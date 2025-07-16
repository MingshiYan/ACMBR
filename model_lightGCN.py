#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2024/8/23 16:16
# @Desc  :
import os.path

import torch
import torch.nn as nn

from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN


class ACMBR(nn.Module):
    def __init__(self, args, dataset):
        super(ACMBR, self).__init__()
        self.device = args.device
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.embedding_size = args.embedding_size
        self.layers = args.layers
        self.behaviors = args.behaviors
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.inter_matrix = dataset.inter_matrix

        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.user_embedding.weight.data[1:])
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embedding.weight.data[1:])

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.cross_loss = nn.BCEWithLogitsLoss()
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_weight = args.reg_weight

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self._construct_graphs()

        self.apply(self._init_weights)

        self._load_model()

    def _construct_graphs(self):
        # 辅助行为图构成
        self.confound_graphs = []
        # 辅助行为和目标行为合并的图
        self.condition_graphs = []
        target_inter_matrix = self.inter_matrix[-1]
        self.target_graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, target_inter_matrix)
        target_inter_matrix_bool = target_inter_matrix.astype(bool)

        for i in range(len(self.behaviors) - 1):
            self.confound_graphs.append(LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, self.inter_matrix[i]))

            tmp_condition_inter_matrix = self.inter_matrix[i].astype(bool)
            tmp_condition_inter_matrix = tmp_condition_inter_matrix + target_inter_matrix_bool
            tmp_target_inter_matrix = tmp_condition_inter_matrix.astype(float)
            tmp_target_inter_matrix = tmp_target_inter_matrix.tocoo()
            self.condition_graphs.append(LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, tmp_target_inter_matrix))

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def multi_grap_operation(self, embs, graphs):
        all_embeddings = []
        for idx in range(len(graphs)):
            tmp_embeddings = graphs[idx](embs)
            all_embeddings.append(tmp_embeddings)
        return all_embeddings

    def min_max_norm(self, input_tensor):

        min_vals = input_tensor.min(dim=0, keepdim=True).values
        max_vals = input_tensor.max(dim=0, keepdim=True).values
        scaled_tensor = (input_tensor - min_vals) / (max_vals - min_vals + 1e-8)
        sum_vals = scaled_tensor.sum(dim=0, keepdim=True)
        normalized_tensor = scaled_tensor / (sum_vals + 1e-8)
        return normalized_tensor

    def forward(self, batch_data):

        user, p_item, n_item = torch.split(batch_data, 1, dim=-1)

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        aux_loss = []
        target_p_confound = []
        target_n_confound = []
        target_p_condition_score = []
        target_n_condition_score = []
        for idx in range(len(self.confound_graphs)):
            # 计算混淆概率
            tmp_confound_embeddings = self.confound_graphs[idx](all_embeddings)
            tmp_confound_user_all_embedding, tmp_confound_item_all_embedding = torch.split(tmp_confound_embeddings, [self.n_users + 1, self.n_items + 1])
            # 计算条件概率
            tmp_condition_embeddings = self.condition_graphs[idx](all_embeddings)
            tmp_condition_user_all_embedding, tmp_condition_item_all_embedding = torch.split(tmp_condition_embeddings, [self.n_users + 1, self.n_items + 1])

            batch_confound_user_emb = tmp_confound_user_all_embedding[user.long()]
            batch_confound_p_item_emb = tmp_confound_item_all_embedding[p_item.long()]
            batch_confound_n_item_emb = tmp_confound_item_all_embedding[n_item.long()]
            p_score = torch.sum(batch_confound_user_emb * batch_confound_p_item_emb, dim=-1).squeeze()
            n_score = torch.sum(batch_confound_user_emb * batch_confound_n_item_emb, dim=-1).squeeze()
            tmp_loss = self.bpr_loss(p_score, n_score)
            aux_loss.append(tmp_loss)

            batch_target_condition_user_emb = tmp_condition_user_all_embedding[user.long()]
            batch_target_condition_p_item_emb = tmp_condition_item_all_embedding[p_item.long()]
            batch_target_condition_n_item_emb = tmp_condition_item_all_embedding[n_item.long()]

            batch_p_condition_score = torch.sum(batch_target_condition_user_emb * batch_target_condition_p_item_emb, dim=-1).squeeze()
            batch_n_condition_score = torch.sum(batch_target_condition_user_emb * batch_target_condition_n_item_emb, dim=-1).squeeze()

            batch_p_confound_score = torch.sum(batch_confound_user_emb * batch_confound_p_item_emb, dim=-1).squeeze()
            batch_n_confound_score = torch.sum(batch_confound_user_emb * batch_confound_n_item_emb, dim=-1).squeeze()

            target_p_condition_score.append(torch.relu(batch_p_condition_score))
            target_n_condition_score.append(torch.relu(batch_n_condition_score))

            target_p_confound.append(torch.relu(batch_p_confound_score))
            target_n_confound.append(torch.relu(batch_n_confound_score))

        aux_loss = torch.stack(aux_loss)
        aux_loss = torch.mean(aux_loss)

        # 考虑因果效应后的得分
        target_p_confound = torch.stack(target_p_confound)
        target_n_confound = torch.stack(target_n_confound)
        target_p_confound = self.min_max_norm(target_p_confound)
        target_n_confound = self.min_max_norm(target_n_confound)

        target_p_condition_score = torch.stack(target_p_condition_score)
        target_n_condition_score = torch.stack(target_n_condition_score)

        rec_p_score = torch.sum(target_p_condition_score * target_p_confound, dim=0)
        rec_n_score = torch.sum(target_n_condition_score * target_n_confound, dim=0)

        rec_loss = self.bpr_loss(rec_p_score, rec_n_score)

        total_loss = rec_loss + self.alpha * aux_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            storage_user_embeddings, storage_item_embeddings = [], []
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

            confound_user_all_embedding = []
            confound_item_all_embedding = []

            condition_user_all_embedding = []
            condition_item_all_embedding = []
            for idx in range(len(self.confound_graphs)):
                # 计算混淆概率
                tmp_confound_embeddings = self.confound_graphs[idx](all_embeddings)
                tmp_confound_user_all_embedding, tmp_confound_item_all_embedding = torch.split(tmp_confound_embeddings, [self.n_users + 1, self.n_items + 1])
                confound_user_all_embedding.append(tmp_confound_user_all_embedding)
                confound_item_all_embedding.append(tmp_confound_item_all_embedding)
                # 计算条件概率
                tmp_condition_embeddings = self.condition_graphs[idx](all_embeddings)
                tmp_condition_user_all_embedding, tmp_condition_item_all_embedding = torch.split(tmp_condition_embeddings, [self.n_users + 1, self.n_items + 1])
                condition_user_all_embedding.append(tmp_condition_user_all_embedding)
                condition_item_all_embedding.append(tmp_condition_item_all_embedding)

            confound_user_all_embedding = torch.stack(confound_user_all_embedding, dim=1)
            confound_item_all_embedding = torch.stack(confound_item_all_embedding, dim=1)

            storage_user_embeddings.append(confound_user_all_embedding)
            storage_item_embeddings.append(confound_item_all_embedding)

            condition_user_all_embedding = torch.stack(condition_user_all_embedding, dim=1)
            condition_item_all_embedding = torch.stack(condition_item_all_embedding, dim=1)

            storage_user_embeddings.append(condition_user_all_embedding)
            storage_item_embeddings.append(condition_item_all_embedding)

            self.storage_user_embeddings = storage_user_embeddings
            self.storage_item_embeddings = storage_item_embeddings

        confound_user_all_embedding, condition_user_all_embedding = self.storage_user_embeddings
        confound_item_all_embedding, condition_item_all_embedding = self.storage_item_embeddings
        confound_user_emb = confound_user_all_embedding[users.long()]
        condition_user_emb = condition_user_all_embedding[users.long()]

        confound_scores = []
        condition_scores = []
        for i in range(len(self.confound_graphs)):
            tmp_confound_user_emb = confound_user_emb[:, i]
            tmp_condition_user_emb = condition_user_emb[:, i]

            tmp_confound_item_emb = confound_item_all_embedding[:, i]
            tmp_condition_item_emb = condition_item_all_embedding[:, i]

            tmp_confound_scores = torch.matmul(tmp_confound_user_emb, tmp_confound_item_emb.t())
            tmp_condition_scores = torch.matmul(tmp_condition_user_emb, tmp_condition_item_emb.t())
            confound_scores.append(torch.relu(tmp_confound_scores))
            condition_scores.append(torch.relu(tmp_condition_scores))

        confound_scores = torch.stack(confound_scores)
        confound_scores = self.min_max_norm(confound_scores)
        condition_scores = torch.stack(condition_scores)
        rec_score = torch.sum(condition_scores * confound_scores, dim=0)

        return rec_score