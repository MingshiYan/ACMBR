# -*- coding: utf-8 -*-

r"""
NCL
################################################

Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import EmbLoss, BPRLoss


class NCL(nn.Module):
    r"""NCL is a neighborhood-enriched contrastive learning paradigm for graph collaborative filtering.
    Both structural and semantic neighbors are explicitly captured as contrastive learning objects.
    """


    def __init__(self, user_count, item_count, inter_matrix, device, embedding_size, layers, ssl_tau=0.1, ssl_weight=1e-4, hyper_layers=1, alpha=1.0, proto_reg=8e-8, num_clusters=1000):

        super(NCL, self).__init__()

        # load dataset info
        self.interaction_matrix = inter_matrix
        self.n_users = user_count
        self.n_items = item_count
        self.device = device

        # load parameters info
        self.latent_dim = embedding_size
        self.n_layers = layers

        self.ssl_temp = ssl_tau
        self.ssl_reg = ssl_weight
        self.hyper_layers = hyper_layers

        self.alpha = alpha
        self.proto_reg = proto_reg
        self.k = num_clusters


        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(self._init_weights)

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def e_step(self, all_embeddings):
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        user_embeddings = user_embeddings.detach().cpu().numpy()
        item_embeddings = item_embeddings.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x"""
        import faiss

        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, all_embeddings):
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            all_embeddings = F.normalize(all_embeddings, dim=-1)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(
            embeddings_list[: self.n_layers + 1], dim=1
        )
        lightgcn_all_embeddings = torch.sum(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(
            node_embedding, [self.n_users, self.n_items]
        )

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]  # [B,]
        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(
            norm_user_embeddings, self.user_centroids.transpose(0, 1)
        )
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).mean()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(
            norm_item_embeddings, self.item_centroids.transpose(0, 1)
        )
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).mean()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.n_users, self.n_items]
        )
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(
            previous_embedding, [self.n_users, self.n_items]
        )

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).mean()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).mean()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def calculate(self, all_embeddings, users, p_items, n_items):
        if self.user_centroids is None:
            self.e_step(all_embeddings)

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward(all_embeddings)

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(
            context_embedding, center_embedding, users, p_items
        )
        proto_loss = self.ProtoNCE_loss(center_embedding, users, p_items)

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[p_items]
        neg_embeddings = item_all_embeddings[n_items]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        return mf_loss + ssl_loss + proto_loss, user_all_embeddings, item_all_embeddings

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[interaction]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores
