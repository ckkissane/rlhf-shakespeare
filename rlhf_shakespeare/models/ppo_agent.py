from torch import nn
from torch.distributions import Categorical


class PpoAgent(nn.Module):
    def __init__(self, logits_net):
        super().__init__()
        self.logits_net = logits_net

    def get_policy(self, obs):
        """
        Args:
            obs: torch.tensor of shape (batch_size, sample_len_so_far)

        Returns:
            dist: Categorical(batch_size, sample_len_so_far, vocab_size)
        """
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def get_cur_action(self, obs):
        """
        Args:
            obs: torch.tensor of shape (batch_size, sample_len_so_far)

        Returns:
            cur_act torch.tensor of shape (), dtype=torch.long
        """
        last_logits = self.logits_net(obs)[:, -1, :]
        policy = Categorical(logits=last_logits)
        return policy.sample()

    def get_cur_logprob(self, obs, cur_act):
        """
        Args:
            obs: torch.tensor of shape (batch_size, sample_len_so_far)
            cur_act: torch.tensor of shape ()

        Returns:
            logp: torch.tensor of shape (), dtype=torch.float32
        """
        last_logits = self.logits_net(obs)[:, -1, :]
        policy = Categorical(logits=last_logits)
        return policy.log_prob(cur_act)
