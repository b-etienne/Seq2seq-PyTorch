import torch


def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


def skip_add_pyramid(x, seq_len, skip_add="add"):
    if len(x.size()) == 2:
        x = x.unsqueeze(0)
    x_len = x.size()[1] // 2
    even = x[:, torch.arange(0, x_len*2-1, 2).long(), :]
    odd = x[:, torch.arange(1, x_len*2, 2).long(), :]
    if skip_add == "add":
        return (even+odd) / 2, ((seq_len) / 2).int()
    else:
        return even, (seq_len / 2).int()




