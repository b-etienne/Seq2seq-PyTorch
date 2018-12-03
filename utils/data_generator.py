import numpy as np
import torch

from torch.utils import data
from random import choice, randrange
from itertools import zip_longest


def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def pad_tensor(vec, pad, value=0, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def pad_collate(batch, values=(0, 0), dim=0):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    # find longest sequence
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch]).int()
    xs = xs[xids]
    ys = ys[yids]
    return xs, ys, sequence_lengths.int(), target_lengths.int()


class ToyDataset(data.Dataset):
    """
    https://talbaumel.github.io/blog/attention/
    """
    def __init__(self, min_length=5, max_length=20, type='train'):
        self.SOS = "<s>"  # all strings will end with the End Of String token
        self.EOS = "</s>"  # all strings will end with the End Of String token
        self.characters = list("abcd")
        self.int2char = list(self.characters)
        self.char2int = {c: i+3 for i, c in enumerate(self.characters)}
        print(self.char2int)
        self.VOCAB_SIZE = len(self.characters)
        self.min_length = min_length
        self.max_length = max_length
        if type == 'train':
            self.set = [self._sample() for _ in range(3000)]
        else:
            self.set = [self._sample() for _ in range(300)]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def _sample(self):
        random_length = randrange(self.min_length, self.max_length)  # Pick a random length
        random_char_list = [choice(self.characters[:-1]) for _ in range(random_length)]  # Pick random chars
        random_string = ''.join(random_char_list)
        a = np.array([self.char2int.get(x) for x in random_string])
        b = np.array([self.char2int.get(x) for x in random_string[::-1]] + [2]) # Return the random string and its reverse
        x = np.zeros((random_length, self.VOCAB_SIZE))

        x[np.arange(random_length), a-3] = 1

        return x, b


