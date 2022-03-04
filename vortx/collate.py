# original vesion from https://github.com/mit-han-lab/torchsparse/blob/master/torchsparse/utils/collate.py
# Copyright (c) 2020-2021 TorchSparse Contributors

# Modified by Noah Stier


import numpy as np
import torch
import torchsparse
import torchsparse.utils


def sparse_collate_tensors(tensors):
    lens = [len(t.C) for t in tensors]
    coords = torch.empty((sum(lens), 4), dtype=torch.int32, device=tensors[0].C.device)
    prev = 0
    for i, n in enumerate(lens):
        coords[prev : prev + n, 3] = i
        coords[prev : prev + n, :3] = tensors[i].C
        prev += n

    feats = torch.cat([t.F for t in tensors], dim=0)
    if feats.dtype is not torch.float32:
        raise Exception("features should be float32")
    return torchsparse.SparseTensor(feats, coords)


def sparse_collate_fn(batch):
    if isinstance(batch[0], dict):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torchsparse.SparseTensor):
                ans_dict[key] = sparse_collate_tensors(
                    [sample[key] for sample in batch]
                )
            elif isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch], axis=0
                )
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.stack([sample[key] for sample in batch], axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn([sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
    else:
        batch_size = batch.__len__()
        ans_dict = tuple()
        for i in range(len(batch[0])):
            key = batch[0][i]
            if isinstance(key, torchsparse.SparseTensor):
                ans_dict += (sparse_collate_tensors([sample[i] for sample in batch]),)
            elif isinstance(key, np.ndarray):
                ans_dict += (
                    torch.stack(
                        [torch.from_numpy(sample[i]) for sample in batch], axis=0
                    ),
                )
            elif isinstance(key, torch.Tensor):
                ans_dict += (torch.stack([sample[i] for sample in batch], axis=0),)
            elif isinstance(key, dict):
                ans_dict += (sparse_collate_fn([sample[i] for sample in batch]),)
            else:
                ans_dict += ([sample[i] for sample in batch],)
        return ans_dict


if __name__ == "__main__":
    batch = []
    for i in range(5):
        n = np.random.randint(100)
        feats = torch.from_numpy(np.random.randn(n).astype(np.float32))
        coords = torch.from_numpy(np.random.randint(100, size=(n, 3)))
        batch.append({"t": torchsparse.SparseTensor(feats, coords)})

    tensors = [b["t"] for b in batch]

    a = torchsparse.utils.sparse_collate_fn(batch)
    b = sparse_collate_fn(batch)

    assert torch.all(a["t"].C == b["t"].C)
    assert torch.all(a["t"].F == b["t"].F)
