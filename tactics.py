import torch
import torch.nn as nn

class TACTiCS():

    def __init__(self):
        pass

    def train(self):
        pass

    def transfer(self):
        pass

class TACTiCSNet(torch.nn.Module):

    def __init__(self, n_classes_A, n_classes_B, n_genes_A, n_genes_B,
                 gene_matches_A, gene_matches_B, mean_A, mean_B, std_A, std_B):
        super(TACTiCSNet, self).__init__()

        self.len_emb = 32
        self.len_final_emb = self.len_emb // 2
        self.n_classes_A, self.n_classes_B = n_classes_A, n_classes_B
        self.n_genes_A, self.n_genes_B = n_genes_A, n_genes_B
        self.gene_matches_A, self.gene_matches_B = gene_matches_A, gene_matches_B
        self.mean_A, self.mean_B = mean_A, mean_B
        self.std_A, self.std_B = std_A, std_B

        self.gene_matches_A = self.gene_matches_A.T  # Q x R --> R x Q
        g_sum_A = self.gene_matches_A.sum(dim=0)  # sum(r) = 1 for r in R
        g_sum_A[g_sum_A == 0] = 1
        self.gene_matches_A = torch.div(self.gene_matches_A, g_sum_A)
        self.gene_matches_A = self.gene_matches_A.T

        g_sum_B = self.gene_matches_B.sum(dim=0)  # sum(q) = 1 for q in Q
        g_sum_B[g_sum_B == 0] = 1
        self.gene_matches_B = torch.div(self.gene_matches_B, g_sum_B)
        self.gene_matches_B = self.gene_matches_B.T # Q x R

        self.relu = nn.ReLU()

        self.register_buffer("mean_A", self.mean_A)
        self.register_buffer("mean_B", self.mean_B)
        self.register_buffer("std_A", self.std_A)
        self.register_buffer("std_B", self.std_B)
        self.register_buffer("gene_matches_A", self.gene_matches_A)
        self.register_buffer("gene_matches_B", self.gene_matches_B)

        self.embed = nn.Linear(in_features=self.n_genes_A + self.n_genes_B, out_features=self.len_emb)

        self.classifier_A = nn.Sequential(
            nn.Linear(self.len_emb, self.len_final_emb),
            nn.ReLU(),
            nn.Linear(self.len_final_emb, self.n_classes_A),
            nn.Softmax(dim=1)
        )

        self.classifier_B = nn.Sequential(
            nn.Linear(self.len_emb, self.len_final_emb),
            nn.ReLU(),
            nn.Linear(self.len_final_emb, self.n_classes_B),
            nn.Softmax(dim=1)
        )

    def forward(self, x_init_A, x_init_B, transfer=False):
        x_init_A_norm = torch.div(torch.add(x_init_A, -self.mean_A), self.std_A)
        x_init_B_norm = torch.div(torch.add(x_init_B, -self.mean_B), self.std_B)

        x_common_A, x_common_B = self.map_genes(x_init_A_norm, x_init_B_norm)
        x_common_A_norm, x_common_B_norm = x_common_A, x_common_B
        x_common_A, x_common_B = self.map_genes(x_init_A, x_init_B)

        # Create embeddings for cells
        x_A = self.embed(x_common_A_norm)
        x_A = self.relu(x_A)
        x_B = self.embed(x_common_B_norm)
        x_B = self.relu(x_B)

        embedding_loss = self.embedding_loss_fn(x_A, x_B)

        if transfer:
            preds_A = self.classifier_B(x_A)
            preds_B = self.classifier_A(x_B)
        else:
            preds_A = self.classifier_A(x_A)
            preds_B = self.classifier_B(x_B)

        return preds_A, preds_B, x_A, x_B, embedding_loss

    def embedding_loss_fn(self, x_A, x_B):
        all_emb = torch.cat([x_A, x_B], dim=0)
        dist_matrix = torch.cdist(all_emb, all_emb)
        n_A = len(x_A)
        n_B = len(x_A)

        dist_n_nbrs = min([self.dist_n_nbrs, n_A - 1, n_B - 1])
        A_A_nn = torch.topk(dist_matrix[:n_A, :n_A], dist_n_nbrs + 1, largest=False, sorted=True).values[:,1:dist_n_nbrs + 1].mean(dim=1)
        A_B_nn = torch.topk(dist_matrix[:n_A, n_A:], dist_n_nbrs, largest=False, sorted=True).values[:,:dist_n_nbrs].mean(dim=1)
        B_B_nn = torch.topk(dist_matrix[n_A:, n_A:], dist_n_nbrs + 1, largest=False, sorted=True).values[:,1:dist_n_nbrs + 1].mean(dim=1)
        B_A_nn = torch.topk(dist_matrix[n_A:, :n_A], dist_n_nbrs, largest=False, sorted=True).values[:,:dist_n_nbrs].mean(dim=1)

        for x in [A_A_nn, A_B_nn, B_B_nn, B_A_nn]:
            x[x == 0] = 1

        embedding_loss_A = torch.div(A_B_nn, A_A_nn)
        embedding_loss_B = torch.div(B_A_nn, B_B_nn)
        embedding_loss = torch.cat([embedding_loss_A, embedding_loss_B], dim=0)
        embedding_loss = embedding_loss.mean()
        embedding_loss = embedding_loss.relu()

        return embedding_loss
