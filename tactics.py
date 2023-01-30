import pandas as pd
import torch
import torch.nn as nn
import utils
import scanpy
import anndata as ad
import pickle
import scipy
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np
import os


class TACTiCS:

    def __init__(self, dict_A=None, dict_B=None, match_path=None, folder=None):
        if folder is not None:
            self.load(folder)
            return

        assert dict_A is not None and dict_B is not None and match_path is not None

        utils.check_species_dict(dict_A)
        utils.check_species_dict(dict_B)
        self.dict_A, self.dict_B = dict_A, dict_B

        with open(match_path, "rb") as f:
            proteins_A, proteins_B, self.gene_matches = pickle.load(f)

        self.adata_A = scanpy.read_h5ad(dict_A["counts"])
        self.adata_B = scanpy.read_h5ad(dict_B["counts"])

        # Remove matches from genes that are not in adata_A or adata_B
        if type(self.gene_matches) is not scipy.sparse.coo_matrix:
            self.gene_matches = scipy.sparse.coo_matrix(self.gene_matches)
        for i in range(len(self.gene_matches.data)):
            if proteins_A[self.gene_matches.row[i]] not in self.adata_A.var_names or \
                    proteins_B[self.gene_matches.col[i]] not in self.adata_B.var_names or \
                    self.gene_matches.data[i] < 0.995:
                self.gene_matches.data[i] = 0
        self.gene_matches.eliminate_zeros()

        # Filter gene matches top-5
        k = 5
        utils.filter_top_k(self.gene_matches, k)

        # Normalize counts and calculate highly variable genes
        self.adata_A = self.norm_adata(self.adata_A, proteins_A)
        self.adata_B = self.norm_adata(self.adata_B, proteins_B)

        # Filter gene matches to highly variable genes + matches
        self.gene_matches = scipy.sparse.coo_matrix(self.gene_matches)
        for i in range(len(self.gene_matches.data)):
            if not self.adata_A.var.highly_variable.loc[proteins_A[self.gene_matches.row[i]]] and \
                    not self.adata_B.var.highly_variable.loc[proteins_B[self.gene_matches.col[i]]]:
                self.gene_matches.data[i] = 0
        self.gene_matches.eliminate_zeros()

        genes_A = sorted([proteins_A[i] for i in set(self.gene_matches.row)])
        genes_B = sorted([proteins_B[i] for i in set(self.gene_matches.col)])

        genes_idx_A = [genes_A.index(x) if x in genes_A else -1 for x in proteins_A]
        genes_idx_B = [genes_B.index(x) if x in genes_B else -1 for x in proteins_B]

        gene_matches_new = torch.zeros((len(genes_A), len(genes_B)), dtype=torch.float32)
        for i in range(len(self.gene_matches.data)):
            gene_matches_new[genes_idx_A[self.gene_matches.row[i]], genes_idx_B[self.gene_matches.col[i]]] = self.gene_matches.data[i]
        self.gene_matches = gene_matches_new

        # Filter counts according to matches
        self.adata_A = self.adata_A[:, genes_A]
        self.adata_B = self.adata_B[:, genes_B]

        # Calculate Z-score per gene
        scanpy.pp.scale(self.adata_A)
        scanpy.pp.scale(self.adata_B)
        self.adata_A.var.drop(["mean", "std"], axis="columns", inplace=True)
        self.adata_B.var.drop(["mean", "std"], axis="columns", inplace=True)

        self.adata_A.raw = self.adata_A
        self.adata_B.raw = self.adata_B

        self.n_classes_A = len(self.adata_A.obs[self.dict_A["column"]].cat.categories)
        self.n_classes_B = len(self.adata_B.obs[self.dict_B["column"]].cat.categories)

        self.model = TACTiCSNet(self.n_classes_A, self.n_classes_B, len(genes_A), len(genes_B), self.gene_matches)
        self.model.apply(utils.init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)

    def train(self, n_epochs=200):
        self.model.train()
        self.model.cuda()

        weight_A = torch.Tensor([1 for _ in range(self.n_classes_A)]).cuda()
        weight_B = torch.Tensor([1 for _ in range(self.n_classes_B)]).cuda()
        loss_fn_A = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight_A)
        loss_fn_B = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight_B)

        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(30):
                self.optimizer.zero_grad()

                x_A, y_A = utils.get_batch(self.adata_A, self.dict_A["column"], 5000)
                x_B, y_B = utils.get_batch(self.adata_B, self.dict_B["column"], 5000)

                x_A, y_A = x_A.cuda(), y_A.cuda()
                x_B, y_B = x_B.cuda(), y_B.cuda()

                preds_A, preds_B, closest_dist = self.model(x_A, x_B)

                preds_top_A = preds_A.argmax(dim=1)
                preds_top_B = preds_B.argmax(dim=1)

                loss = loss_fn_A(preds_A, y_A) + loss_fn_B(preds_B, y_B) + closest_dist
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()

            utils.update_weights_focal(y_A, preds_top_A, self.n_classes_A, weight_A)
            utils.update_weights_focal(y_B, preds_top_B, self.n_classes_B, weight_B)

    def transfer(self):
        self.model.eval()
        self.model.cpu()

        # Transfer cell types and embed cells
        x_A, y_A = utils.get_batch(self.adata_A, self.dict_A["column"], len(self.adata_A))
        preds_A, emb_A = self.model.transfer_A(x_A)
        preds_top_A = preds_A.argmax(dim=1).numpy()

        x_B, y_B = utils.get_batch(self.adata_B, self.dict_B["column"], len(self.adata_B))
        preds_B, emb_B = self.model.transfer_B(x_B)
        preds_top_B = preds_B.argmax(dim=1).numpy()

        y_A, y_B = y_A.numpy(), y_B.numpy()

        # Store results
        self.adata_A.obs[self.dict_B["name"]] = pd.Categorical.from_codes(preds_top_A, dtype=self.adata_B.obs[self.dict_B["column"]].dtype)
        self.adata_B.obs[self.dict_A["name"]] = pd.Categorical.from_codes(preds_top_B, dtype=self.adata_A.obs[self.dict_A["column"]].dtype)

        self.adata_A.obsm["emb"] = emb_A.numpy()
        self.adata_B.obsm["emb"] = emb_B.numpy()

    def norm_adata(self, adata: ad.AnnData, proteins):
        # Normalize per cell and with log
        scanpy.pp.normalize_total(adata, 10000)
        scanpy.pp.log1p(adata)
        del adata.uns["log1p"]

        # Filter matrix to genes with protein sequences
        adata = adata[:, adata.var_names.isin(proteins)]

        # Calculate highly variable genes
        scanpy.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata.var.drop(["means", "dispersions", "dispersions_norm"], axis="columns", inplace=True)
        del adata.uns["hvg"]
        return adata

    def plot_matches(self):
        plt.rcParams["savefig.bbox"] = "tight"

        labels_A, labels_B, matrix = self.get_avg_conf()
        sns.heatmap(matrix, cmap="PuRd", linecolor="white", linewidths=0.1, vmin=0, vmax=1, square=True,
                    cbar=True)
        plt.xticks([x + 0.5 for x in range(len(labels_B))], labels_B, rotation=-90)
        plt.yticks([x + 0.5 for x in range(len(labels_A))], labels_A, rotation=0)
        plt.xlabel(self.dict_B["name"])
        plt.ylabel(self.dict_A["name"])
        plt.show()

    def plot_embeddings(self):
        new_X = np.concatenate([self.adata_A.obsm["emb"], self.adata_B.obsm["emb"]], axis=0)
        adata = ad.AnnData(new_X)
        adata.obs["Species"] = [self.dict_A["name"] for _ in range(len(self.adata_A))] + \
                               [self.dict_B["name"] for _ in range(len(self.adata_B))]
        adata.obs["Cell type"] = np.concatenate([self.adata_A.obs[self.dict_A["column"]].to_numpy(),
                                 self.adata_B.obs[self.dict_B["column"]].to_numpy()], axis=0)

        scanpy.pp.neighbors(adata, n_neighbors=30)
        scanpy.tl.umap(adata)
        scanpy.pl.umap(adata, color=["Species", "Cell type"], frameon=False)

    def save(self, folder=None):
        if folder is None:
            folder = self.dict_A["name"] + "_" + self.dict_B["name"]
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.adata_A.write_h5ad(os.path.join(folder, "species_A_pr.h5ad"), compression="lzf")
        self.adata_B.write_h5ad(os.path.join(folder, "species_B_pr.h5ad"), compression="lzf")

        with open(os.path.join(folder, "matches.pkl"), "wb") as f:
            pickle.dump([self.dict_A, self.dict_B, self.gene_matches], f)

        torch.save(self.model.state_dict(), os.path.join(folder, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(folder, "optim.pth"))

    def load(self, folder):
        assert os.path.exists(folder)

        self.adata_A = ad.read_h5ad(os.path.join(folder, "species_A_pr.h5ad"))
        self.adata_B = ad.read_h5ad(os.path.join(folder, "species_B_pr.h5ad"))

        with open(os.path.join(folder, "matches.pkl"), "rb") as f:
            self.dict_A, self.dict_B, self.gene_matches = pickle.load(f)

        self.n_classes_A = len(self.adata_A.obs[self.dict_A["column"]].cat.categories)
        self.n_classes_B = len(self.adata_B.obs[self.dict_B["column"]].cat.categories)

        self.model = TACTiCSNet(self.n_classes_A, self.n_classes_B, len(self.adata_A.var_names), len(self.adata_B.var_names), self.gene_matches)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)

        self.model.load_state_dict(torch.load(os.path.join(folder, "model.pth")))
        self.optimizer.load_state_dict(torch.load(os.path.join(folder, "optim.pth")))

    def get_avg_conf(self):
        labels_A = set(self.adata_A.obs[self.dict_A["column"]].cat.categories)
        labels_B = set(self.adata_B.obs[self.dict_B["column"]].cat.categories)
        labels = np.array(sorted(list(labels_A.union(labels_B))))
        idx_A = [i for i in range(len(labels)) if labels[i] in set(labels_A)]
        idx_B = [i for i in range(len(labels)) if labels[i] in set(labels_B)]
        labels_A = labels[idx_A]
        labels_B = labels[idx_B]

        y_A = self.adata_A.obs[self.dict_A["column"]].values
        transfer_A = self.adata_A.obs[self.dict_B["name"]].values
        y_B = self.adata_B.obs[self.dict_B["column"]].values
        transfer_B = self.adata_B.obs[self.dict_A["name"]].values

        conf_matrix_A = sklearn.metrics.confusion_matrix(y_A, transfer_A, normalize="true", labels=labels)
        conf_matrix_A = conf_matrix_A[idx_A, :]
        conf_matrix_A = conf_matrix_A[:, idx_B]

        conf_matrix_B = sklearn.metrics.confusion_matrix(y_B, transfer_B, normalize="true", labels=labels)
        conf_matrix_B = conf_matrix_B[idx_B, :]
        conf_matrix_B = conf_matrix_B[:, idx_A]

        conf_matrix_AB = np.mean([conf_matrix_A, conf_matrix_B.T], axis=0)
        return labels_A, labels_B, conf_matrix_AB

    def get_ADS(self):
        labels_A, labels_B, matrix = self.get_avg_conf()
        idx_A = [labels_A[i] in labels_B for i in range(len(labels_A))]
        idx_B = [labels_B[i] in labels_A for i in range(len(labels_B))]

        matrix = matrix[idx_A, :]
        matrix = matrix[:, idx_B]

        print("ADS:", matrix.diagonal().mean())

    def get_recall(self):
        labels_A, labels_B, matrix = self.get_avg_conf()
        correct = 0
        total = len(set(labels_A).intersection(labels_B))
        for i in range(len(labels_A)):
            if labels_A[i] not in labels_B:
                continue
            j = np.where(labels_B == labels_A[i])
            if matrix[i, j] == matrix[i, :].max() and matrix[i, j] == matrix[:, j].max():
                correct += 1

        if total == 0:
            print("No common cell types")
            return

        print("Recall:", correct/total, f"{correct}/{total}")


class TACTiCSNet(torch.nn.Module):

    def __init__(self, n_classes_A, n_classes_B, n_genes_A, n_genes_B, gene_dist):
        super(TACTiCSNet, self).__init__()
        self.n_nbrs = 20

        self.linear_cell = nn.Sequential(nn.Linear(in_features=n_genes_A + n_genes_B, out_features=64),
                                         nn.ReLU(),
                                         nn.Linear(in_features=64, out_features=32),
                                         nn.ReLU())
        self.classifier_A = nn.Linear(in_features=32, out_features=n_classes_A)
        self.classifier_B = nn.Linear(in_features=32, out_features=n_classes_B)

        # Create normalization terms for gene weights
        self.gene_dist = gene_dist
        self.gene_dist_norm_A = self.gene_dist.sum(dim=1)
        self.gene_dist_norm_A[self.gene_dist_norm_A == 0] = 1
        self.gene_dist_norm_B = self.gene_dist.sum(dim=0)
        self.gene_dist_norm_B[self.gene_dist_norm_B == 0] = 1

    def cuda(self, device=None):
        self.gene_dist = self.gene_dist.cuda(device)
        self.gene_dist_norm_A = self.gene_dist_norm_A.cuda(device)
        self.gene_dist_norm_B = self.gene_dist_norm_B.cuda(device)
        return super(TACTiCSNet, self).cuda(device)

    def cpu(self):
        self.gene_dist = self.gene_dist.cpu()
        self.gene_dist_norm_A = self.gene_dist_norm_A.cpu()
        self.gene_dist_norm_B = self.gene_dist_norm_B.cpu()
        return super(TACTiCSNet, self).cpu()

    def forward(self, x_A, x_B, train=True):
        # Calculate shared feature space by imputing and concatenating expression
        x_impute_A = torch.matmul(x_A, self.gene_dist)
        x_impute_A = torch.div(x_impute_A, self.gene_dist_norm_B)
        x_shared_A = torch.cat([x_A, x_impute_A], dim=1)

        x_impute_B = torch.matmul(x_B, self.gene_dist.T)
        x_impute_B = torch.div(x_impute_B, self.gene_dist_norm_A)
        x_shared_B = torch.cat([x_impute_B, x_B], dim=1)

        # Embed cells
        x_emb_A = self.linear_cell(x_shared_A)
        x_emb_B = self.linear_cell(x_shared_B)

        # Predict cell type probabilities
        x_preds_A = self.classifier_A(x_emb_A)
        x_preds_B = self.classifier_B(x_emb_B)

        dist_matrix = torch.cdist(x_emb_A, x_emb_B)
        n_nbrs = min([self.n_nbrs, len(x_emb_A), len(x_emb_B)])

        nn_A = torch.topk(dist_matrix, n_nbrs, largest=False, dim=1).indices
        nn_B = torch.topk(dist_matrix, n_nbrs, largest=False, dim=0).indices.T

        pred_A = torch.nn.functional.embedding_bag(nn_A, x_impute_B, mode="mean")
        pred_B = torch.nn.functional.embedding_bag(nn_B, x_impute_A, mode="mean")

        emb_loss = torch.nn.functional.mse_loss(pred_A, x_A) + torch.nn.functional.mse_loss(pred_B, x_B)
        return x_preds_A, x_preds_B, emb_loss

    def transfer_A(self, x_A):
        with torch.no_grad():
            # Create shared feature space
            x_impute_A = torch.matmul(x_A, self.gene_dist)
            x_impute_A = torch.div(x_impute_A, self.gene_dist_norm_B)
            x_shared_A = torch.cat([x_A, x_impute_A], dim=1)

            # Embed cells
            x_emb_A = self.linear_cell(x_shared_A)
            x_preds_A = self.classifier_B(x_emb_A)
            return x_preds_A, x_emb_A

    def transfer_B(self, x_B):
        with torch.no_grad():
            # Create shared feature space
            x_impute_B = torch.matmul(x_B, self.gene_dist.T)
            x_impute_B = torch.div(x_impute_B, self.gene_dist_norm_A)
            x_shared_B = torch.cat([x_impute_B, x_B], dim=1)

            # Embed cells
            x_emb_B = self.linear_cell(x_shared_B)
            x_preds_B = self.classifier_A(x_emb_B)
            return x_preds_B, x_emb_B
