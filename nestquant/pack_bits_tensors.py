"""
Pack-bits Tensors
From: Felix Petersen and Tobias Sutter, Distributional Quantization, 2023
https://github.com/Felix-Petersen/distquant
"""

import torch
import math

# pack the simulated bits tensors with unsigned INT64 data type
def pack_simulated_bits_tensor(q_tensor, k=3, n=None, packbits=True, signed=True):
    if signed:
        q_tensor = q_tensor + 2 ** (k - 1)
    # k is number of bits to quantize to; alternatively specify `n`.
    # n is number of quantization points; mutually exclusive to `k`.
    assert n is not None or k is not None, '`k` bits or `n` values needs to be specified.'
    assert not (n is not None and k is not None), 'only one of `k` bits or `n` values can be specified.'
    if n is None:
        n = 2 ** k
    shape = list(q_tensor.shape)
    q_tensor = q_tensor.view(-1)
    # signed INT convert to unsigned INT

    if packbits:
        if k is None:
            k = math.ceil(math.log2(n))
        num_per_uint64 = 64 // k
        q_tensor = torch.cat([q_tensor, torch.zeros((num_per_uint64 - (q_tensor.shape[0] % num_per_uint64)) % num_per_uint64,
                                                    device=q_tensor.device, dtype=torch.int64)])
        q_tensor = q_tensor.reshape(-1, num_per_uint64)
        packed_tensor = torch.zeros(q_tensor.shape[0], device=q_tensor.device, dtype=torch.int64)
        for i in range(num_per_uint64):
            packed_tensor <<= k
            packed_tensor |= q_tensor[:, i]
        q_tensor = packed_tensor
    else:
        q_tensor = q_tensor.view(*shape)

    return q_tensor


# unpack the unsigned INT64 data type pack bits tensors
def unpack_bits_tensor(q_tensor, k=3, n=None, shape=None, signed=True):
    assert shape is not None
    assert n is not None or k is not None
    if k is None:
        k = math.ceil(math.log2(n))
    n_elems = math.prod(list(shape))
    num_per_uint64 = 64 // k
    unpacked_tensor = torch.zeros(q_tensor.shape[0], num_per_uint64, device=q_tensor.device,
                                  dtype=torch.int64)
    mask_tensor = torch.zeros(1, device=q_tensor.device, dtype=torch.int64) | (2 ** k - 1)
    for i in range(num_per_uint64 - 1, -1, -1):
        unpacked_tensor[:, i] = q_tensor & mask_tensor
        q_tensor >>= k
    q_tensor = unpacked_tensor.view(-1)[:n_elems]
    q_tensor = q_tensor.view(*shape)
    # unsigned INT convert to signed INT
    if signed:
        q_tensor = q_tensor - 2 ** (k - 1)
    return q_tensor
