import torch
import torch.nn.functional as F
from datasets import WordDataset
from models.pretrained_model import DecoderOnlyTransformer
from models.reward_model import RewardModel
from models.ppo_agent import PpoAgent
import os
import copy
import numpy as np
import sys


# standard ppo loss fn
def compute_pi_loss(
    orig_ppo_agent, ppo_agent, old_log_probs, obs, act, weights, clip_ratio=0.2
):
    """
    Args:
        orig_ppo_agent: PpoAgent
        ppo_agent: PpoAgent
        old_log_probs: torch.tensor of shape (batch_size, sample_len)
        obs: torch.tensor of shape (batch_size, sample_len)
        act: torch.tensor of shape (batch_size, sample_len)
        weights: torch.tensor of shape (batch_size, sample_len)
        clip_ratio: float

    Returns:
        loss_pi: torch.tensor of shape ()
    """
    old_logp = old_log_probs
    logp = ppo_agent.get_policy(obs).log_prob(act)
    ratio = torch.exp(logp - old_logp)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * weights
    loss_pi = -(torch.min(ratio * weights, clip_adv)).mean()

    # info for debugging
    clipped = ratio.gt(1.0 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    orig_logp = orig_ppo_agent.get_policy(obs).log_prob(act)
    approx_kl = (0.5 * (logp - orig_logp) ** 2).mean().item()

    pi_info = dict(kl=approx_kl, cf=clipfrac)

    return loss_pi, pi_info


def train_ppo(
    pi_lr=3e-5,
    max_iters=100,
    batch_size=100,
    train_pi_iters=4,
    sample_len=64,
    num_eval_samples=3,
    shakespeare_corpus_path="./data/100-0.txt",
):

    print(f"pi_lr: {pi_lr}")
    print(f"max_iters: {max_iters}")
    print(f"batch_size: {batch_size}")
    print(f"train_pi_iters: {train_pi_iters}")
    print(f"sample_len: {sample_len}")
    print(f"num_eval_samples: {num_eval_samples}")
    print(f"shakespeare_corpus_path: {shakespeare_corpus_path}")

    # Load this so we ensure we use same vocab
    text = open(shakespeare_corpus_path, "r", encoding="utf-8-sig").read()
    block_size = 128  # TODO: don't hardcode this
    word_dataset = WordDataset(text, block_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load pretrained model
    # TODO: don't harcode this config
    pretrained_model = (
        DecoderOnlyTransformer(
            num_layers=8,
            num_heads=8,
            vocab_size=word_dataset.vocab_size,
            hidden_size=512,
            max_pos_embeddings=word_dataset.block_size,
            dropout=0.0,
        )
        .to(device)
        .train()
    )

    pretrained_ckpt_path = os.path.join(os.getcwd(), "checkpoints/model.pt")
    pretrained_model.load_state_dict(
        torch.load(pretrained_ckpt_path, map_location=device)
    )

    # reference to compare KL(current_model||original_model)
    orig_ppo_agent = PpoAgent(pretrained_model).to(device).eval()

    # init policy
    pi_model = copy.deepcopy(pretrained_model).to(device).train()
    ppo_agent = PpoAgent(pi_model).to(device).train()
    pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=pi_lr)

    # load reward model
    rew_model = RewardModel(pretrained_model, num_classes=2).to(device).train()
    rew_ckpt_path = os.path.join(os.getcwd(), "checkpoints/reward_model.pt")
    rew_model.load_state_dict(torch.load(rew_ckpt_path, map_location=device))
    rew_model.eval()

    # Initial Eval (before training) to compare
    ppo_agent.eval()
    with torch.no_grad():
        context = "\n"  # unconditional sample
        x = torch.tensor([word_dataset.stoi[context]], dtype=torch.long)[None, ...].to(
            device
        )
        for _ in range(sample_len - 1):
            next_token = ppo_agent.get_cur_action(x)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=-1)
        completion = "".join([word_dataset.itos[int(i)] for i in x.squeeze()])
        print("initial sample (before training):", completion[: completion.rfind("\n")])
    ppo_agent.train()

    # Training loop
    for it in range(max_iters):
        ptr = 0
        batch_obs = torch.zeros(
            (batch_size, sample_len), dtype=torch.long, device=device
        )
        batch_acts = torch.zeros(
            (batch_size, sample_len), dtype=torch.long, device=device
        )
        batch_weights = torch.zeros((batch_size, sample_len), device=device)
        batch_rets = torch.zeros((batch_size,))
        batch_log_probs = torch.zeros((batch_size, sample_len), device=device)

        with torch.no_grad():
            # init obs / act / log_prob for unconditional sample
            context = "\n"
            obs = torch.tensor([word_dataset.stoi[context]], dtype=torch.long)[
                None, ...
            ].to(device)
            act = torch.tensor([[]], dtype=torch.long).to(device)

            while True:
                cur_act = ppo_agent.get_cur_action(obs)
                act = torch.cat((act, cur_act.unsqueeze(0)), dim=-1)

                done = act.shape[-1] >= sample_len
                if done:
                    # store obs, acts, lob_probs
                    batch_obs[ptr] = obs
                    batch_acts[ptr] = act
                    log_prob = ppo_agent.get_policy(obs).log_prob(act)
                    batch_log_probs[ptr] = log_prob.squeeze()

                    # get reward for this obs sample
                    padding_token = word_dataset.stoi["\n"]
                    obs_pad = F.pad(
                        obs, (block_size - obs.numel(), 0), "constant", padding_token
                    )
                    rew_logits = rew_model(obs_pad)[:, -1, :]
                    rew_scores = rew_logits.squeeze()
                    rew_probs = rew_scores.softmax(dim=-1)
                    prob_happy = rew_probs[1]
                    ep_ret = torch.log(prob_happy)

                    batch_rets[ptr] = ep_ret
                    batch_weights[ptr] = ep_ret.item() * torch.ones_like(log_prob)

                    # Reset obs / act / done / for next iter
                    context = "\n"
                    obs = torch.tensor([word_dataset.stoi[context]], dtype=torch.long)[
                        None, ...
                    ].to(device)
                    done = False
                    act = torch.tensor([[]], dtype=torch.long).to(device)
                    ptr += 1
                    if ptr >= batch_size:
                        break
                else:
                    obs = torch.cat((obs, cur_act.unsqueeze(0)), dim=-1)

        # adv normalization trick
        batch_weights = (batch_weights - batch_weights.mean()) / batch_weights.std()
        # optimization step(s)
        for _ in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss, pi_info = compute_pi_loss(
                orig_ppo_agent,
                ppo_agent,
                batch_log_probs,
                batch_obs,
                batch_acts,
                batch_weights,
            )
            loss.backward()
            pi_optimizer.step()

        print(f"\nit: {it}, avg ret: {torch.mean(batch_rets)}")
        print(f"clipfrac: {pi_info['cf']}")
        print(f"KL(cur_policy||orig_policy): {pi_info['kl']}")
        print(f"sqrt KL: {np.sqrt(pi_info['kl'])}")

        # Eval
        ppo_agent.eval()
        with torch.no_grad():
            for i in range(num_eval_samples):
                context = "\n"
                x = torch.tensor([word_dataset.stoi[context]], dtype=torch.long)[
                    None, ...
                ].to(device)
                for _ in range(sample_len - 1):
                    next_token = ppo_agent.get_cur_action(x)
                    x = torch.cat((x, next_token.unsqueeze(0)), dim=-1)
                # sanity check reward for this sample
                with torch.no_grad():
                    padding_token = word_dataset.stoi["\n"]
                    x_pad = F.pad(
                        x, (block_size - x.numel(), 0), "constant", padding_token
                    )
                    rew_logits = rew_model(x_pad)[:, -1, :]
                    rew_scores = rew_logits.squeeze()
                    rew_probs = rew_scores.softmax(dim=-1)
                # check the sample
                completion = "".join([word_dataset.itos[int(i)] for i in x.squeeze()])
                print("\neval sample ", i, ":", completion[: completion.rfind("\n")])
                print("\nrew_probs", rew_probs)
                prob_happy = rew_probs[1]
                print("prob_happy", prob_happy)
                ret = torch.log(prob_happy)
                print("return", ret)
        # Save checkpoint
        ppo_agent.train()
        ckpt_path = os.path.join(
            os.getcwd(), "checkpoints/policy_it_" + str(it) + ".pt"
        )
        print("saving policy")
        torch.save(pi_model.state_dict(), ckpt_path)
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pi_lr", type=float, default=3e-5)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--train_pi_iters", type=int, default=4)
    parser.add_argument("--sample_len", type=int, default=64)
    parser.add_argument("--num_eval_samples", type=int, default=3)
    parser.add_argument(
        "--shakespeare_corpus_path", type=str, default="./data/100-0.txt"
    )
    args = parser.parse_args()

    train_ppo(
        pi_lr=args.pi_lr,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        train_pi_iters=args.train_pi_iters,
        sample_len=args.sample_len,
        num_eval_samples=args.num_eval_samples,
        shakespeare_corpus_path=args.shakespeare_corpus_path,
    )
