import numpy as np
import torch
import pandas as pd
import scipy


def check_species_dict(species_dict):
    assert "name" in species_dict, "name is missing from dictionary"
    assert "column" in species_dict, "column is missing from dictionary"


def update_weights_focal(true, preds, n_classes, weight):
    acc_dict = {}
    for i in range(len(true)):
        y = true[i].item()
        if y not in acc_dict:
            acc_dict[y] = [0, 0]
        if y == preds[i].item():
            acc_dict[y][0] += 1
        acc_dict[y][1] += 1
    
    total_true = sum(acc_dict[i][0] for i in acc_dict)
    acc = total_true / len(true)
    for i in range(n_classes):
        if i in acc_dict:
            weight[i] = (1 - max(acc_dict[i][0] / acc_dict[i][1], 1)) * 9
            weight[i] *= acc
            weight[i] += 1


def get_batch(adata, column, n):
    total = len(adata)
    if n == total:
        x = torch.Tensor(adata.X)
        y = adata.obs[column].cat.codes
    else:
        weights = 1 / adata.obs[column].value_counts()
        cts = adata.obs[column]
        weights = np.array(pd.Series.replace(cts, weights).values)
        weights = weights / sum(weights)
        idx = sorted(np.random.choice(total, n, replace=True, p=weights))

        x = torch.Tensor(adata[idx, :].X)
        y = adata.obs[column].cat.codes[idx]
    y = torch.Tensor(y).to(torch.long)
    return x, y


def filter_top_k(matrix, k, dim=None):
    if dim is None:
        matrix = scipy.sparse.csr_matrix(matrix)  # row-wise
        filter_top_k(matrix, k, 0)
        matrix = scipy.sparse.csc_matrix(matrix)  # column-wise
        filter_top_k(matrix, k, 1)
        return

    for i in range(matrix.shape[dim]):  # for every row
        values = matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]]
        if len(values) == 0:  # gene has no matches
            continue
        else:  # gene has at least one match meeting threshold
            if len(values) > k:
                min_value = sorted(values, reverse=True)[
                    k]  # get the value of the k+1th item, so we only take indices > value: enforces a strict maximum.
                values[values <= min_value] = 0
    matrix.eliminate_zeros()


def init_weights(m):
    with torch.no_grad():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.fill_(0)
