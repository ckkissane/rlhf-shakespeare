import torch
import torch.nn.functional as F
from datasets import WordDataset, SentimentDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.pretrained_model import DecoderOnlyTransformer
from models.reward_model import RewardModel
import os


def train_rew_model(
    batch_size=8,
    sample_len=64,
    max_epochs=1,
    lr=1e-4,
    human_data_path="./data/handcrafted_data.jsonl",
    shakespeare_corpus_path="./data/100-0.txt",
):
    """Fine-tunes a reward model to classify shakespeare samples with positve sentiment"""

    print(f"batch_size: {batch_size}")
    print(f"sample_len: {sample_len}")
    print(f"max_epochs: {max_epochs}")
    print(f"lr: {lr}")
    print(f"human_data_path: {human_data_path}")
    print(f"shakespeare_corpus_path: {shakespeare_corpus_path}")

    # note: these must be the same as you used in pre-training
    text = open(shakespeare_corpus_path, "r", encoding="utf-8-sig").read()
    block_size = 128  # TODO: don't hardcode this
    word_dataset = WordDataset(text, block_size)

    train_dataset = SentimentDataset(
        human_data_path,
        word_dataset.stoi,
        word_dataset.block_size,
        sample_len=sample_len,
        split="train",
    )
    print("sentiment train_dataset size", len(train_dataset))

    test_dataset = SentimentDataset(
        human_data_path,
        word_dataset.stoi,
        word_dataset.block_size,
        sample_len=sample_len,
        split="test",
    )
    print("sentiment test_dataset size", len(test_dataset))

    train_loader = DataLoader(
        train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
    )
    test_loader = DataLoader(
        test_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # TODO: don't hardcode this config
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

    rew_model = RewardModel(pretrained_model, num_classes=2).to(device).train()
    optimizer = torch.optim.Adam(rew_model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, (x, y) in pbar:
            logits = rew_model(x)

            optimizer.zero_grad()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-100
            )
            loss.backward()
            optimizer.step()

            pbar.set_description(f"epoch: {epoch}, it: {it}, loss:{loss.item():.5f}")

        # eval
        rew_model.eval()
        correct, total = 0, 0
        for test_x, test_y in test_loader:
            test_logits = rew_model(test_x)
            pred = test_logits.argmax(dim=-1)
            correct += (pred == test_y).sum()
            total += test_x.shape[0]
        print(f"validation accuracy: {correct / total}")

        # save model
        print("saving reward model")
        ckpt_path = os.path.join(os.getcwd(), "checkpoints/reward_model.pt")
        torch.save(rew_model.state_dict(), ckpt_path)
        rew_model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_len", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--human_data_path", type=str, default="./data/handcrafted_data.jsonl"
    )
    parser.add_argument(
        "--shakespeare_corpus_path", type=str, default="./data/100-0.txt"
    )
    args = parser.parse_args()

    train_rew_model(
        batch_size=args.batch_size,
        sample_len=args.sample_len,
        max_epochs=args.max_epochs,
        lr=args.lr,
        human_data_path=args.human_data_path,
        shakespeare_corpus_path=args.shakespeare_corpus_path,
    )
