import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import re
import jsonlines
import random


class WordDataset(Dataset):
    """
    arrange data and targets so that the first i elements of x
    will be asked to predict the i-th element of y. Notice that
    the eventual language model will actually make block_size
    individual predictions at the same time based on this data,
    so we are being clever and amortizing the cost of the forward
    pass of the network. So for example if block_size is 4, then
    we could e.g. sample a chunk of text "w1 w2 w3 w4 w5", the integers in
    x will correspond to "w1 w2 w3 w4" and in y will be "w2 w3 w4 w5". This will
    then actually "multitask" 4 separate examples at the same time
    in the language model:
    - given just "w1", please predict "w2" as next
    - given "w1 w2" please predict "w3" next
    - given "w1 w2 w3" predict "w4" next
    - given "w1 w2 w3 w4" predict "w5" next
    """

    def __init__(self, data, block_size):
        words = re.split(r"\b", data)
        vocab = sorted(list(set(words)))
        data_size, vocab_size = len(words), len(vocab)
        print("data has %d words, %d unique." % (data_size, vocab_size))

        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = words

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[
            idx * self.block_size : (idx * self.block_size) + self.block_size + 1
        ]
        # encode every word to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, word_to_idx, block_size, sample_len, split):
        self.word_to_idx = (
            word_to_idx  # important: must use same mapping as pre-trained model
        )
        self.block_size = block_size
        self.sample_len = sample_len
        objs = []
        with jsonlines.open(path_to_data) as reader:
            for obj in reader:
                objs.append(obj)

        self.samples = []
        self.labels = []
        random.shuffle(objs)
        for obj in objs:
            self.samples.append(obj["sample"])
            self.labels.append(obj["sentiment"])
        total_samples = len(self.samples)
        num_test = int(total_samples * 0.2)
        self.samples = (
            self.samples[:num_test] if split == "test" else self.samples[num_test:]
        )
        self.labels = (
            self.labels[:num_test] if split == "test" else self.labels[num_test:]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        label = self.labels[item]

        input_ids = [self.word_to_idx[word] for word in re.split(r"\b", sample)]
        input_ids = input_ids[: self.sample_len]  # sample might be too long
        x = torch.tensor(input_ids, dtype=torch.long)

        padding_token = self.word_to_idx["\n"]  # TODO: not sure if this should work...
        x = F.pad(x, (self.block_size - x.numel(), 0), "constant", padding_token)

        y = -100 * torch.ones_like(x)
        y[-1] = 1 if label == "happy" else 0

        return x, y
