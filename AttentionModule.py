

import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.init as init
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size, bias=False)

        self.att_dropout = nn.Dropout(0.2)
        self.Bilinear_att = nn.Linear(self.att_size, self.att_size, bias=False)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        """
        q (target_rel):  (few/b, 1, dim)
        k (nbr_rel):    (few/b, max, dim)
        v (nbr_ent):    (few/b, max, dim)
        mask:   (few/b, max)
        output:
        """
        q = q.unsqueeze(1)
        orig_q_size = q.size()
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, 1, num_heads, att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, max, num_heads, att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)   #(few/b, max, num_heads, att_size)


        q = q.transpose(1, 2)  #(few/b, num_heads, 1, att_size)
        k = k.transpose(1, 2).transpose(2, 3)  #(few/b, num_heads, att_size, max)
        v = v.transpose(1, 2)  #(few/b, num_heads, max, att_size)

        x = torch.matmul(self.Bilinear_att(q), k)

        x = torch.softmax(x, dim=3)   # [few/b, num_heads, 1, max]

        x = self.att_dropout(x)     # [few/b, num_heads, 1, max]
        x = x.matmul(v)    #(few/b, num_heads, 1, att_size)

        x = x.transpose(1, 2).contiguous()  # (few/b, 1, num_heads, att_size)

        x = x.view(batch_size, -1, self.num_heads * self.att_size).squeeze(1) #(few/b, dim)
        x = self.output_layer(x)  #(few/b, dim)

        return x



class Attention_Module(nn.Module):
    def __init__(self, args, embed, num_symbols, embedding_size, use_pretrain=True, finetune=True, dropout_rate=0.3):
        super(Attention_Module, self).__init__()

        self.agrs = args
        self.embedding_size = embedding_size
        self.pad_idx = num_symbols
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(args.device)

        self.symbol_emb = nn.Embedding(num_symbols+1, self.embedding_size, padding_idx=self.pad_idx)
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_MLP = nn.Dropout(0.2)

        self.LeakyRelu = nn.LeakyReLU()
        self.task_aware_attention_module = Attention(hidden_size=self.embedding_size, num_heads=1) #
        self.entity_pair_attention_module = Attention(hidden_size=self.embedding_size, num_heads=1) #

        self.layer_norm = nn.LayerNorm(self.embedding_size)


        self.gate_w = nn.Linear(2*self.embedding_size, self.embedding_size) #

        self.Linear_tail = nn.Linear(self.embedding_size, self.embedding_size, bias=False) #
        self.Linear_head = nn.Linear(self.embedding_size, self.embedding_size, bias=False)


        self.rel_w = nn.Bilinear(self.embedding_size, self.embedding_size, 2* self.embedding_size, bias=False) #2*

        self.MLPW1 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size)

        self.MLPW2 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size, bias=False)

        init.xavier_normal_(self.MLPW1.weight)
        init.xavier_normal_(self.MLPW2.weight)
        init.xavier_normal_(self.gate_w.weight)

        self.layer_norm1 = nn.LayerNorm(2 * self.embedding_size)

        # 门控机制参数
        self.W_g = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.U_g = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.b_g = nn.Parameter(torch.zeros(self.embedding_size))

        self.W_r = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.U_r = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.b_r = nn.Parameter(torch.zeros(self.embedding_size))

        #门控2参数
        self.W_g1 = nn.Linear(2*self.embedding_size,1)
        self.W_g2 = nn.Linear(2 * self.embedding_size, 1)
        self.W_g3 = nn.Linear(2 * self.embedding_size, 1)
        self.W_g4 = nn.Linear(2 * self.embedding_size, 1)

        #门控3
        self.W_g5 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.U_g5 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.b_g5 = nn.Parameter(torch.zeros(self.embedding_size))


    def bi_MLP(self, input):
        output = torch.relu(self.MLPW1(input))
        output = self.MLPW2(output)
        output = self.layer_norm1(output + input)
        return output

    # def softmax(self,x):
    #     """Softmax 函数"""
    #     exp_x = np.exp(x - np.max(x))  # 防止溢出
    #     return exp_x / np.sum(exp_x)

    def compute_attention_weights(self,h, h_s, r):
        """
        使用点积注意力机制计算h和h_s对r的重要性权重
        参数:
        h (numpy array): 向量h
        h_s (numpy array): 向量h_s
        r (numpy array): 向量r

        返回:
        tuple: 归一化后的权重 (weight_h, weight_h_s)
        """

        score_h = torch.sum(h * r, dim=1)

        score_h_s = torch.sum(h_s * r, dim=1)

        scores = torch.stack([score_h, score_h_s], dim=1)

        weights = F.softmax(scores, dim=1)

        weight_h = weights[:, 0]
        weight_h_s = weights[:, 1]


        return weight_h, weight_h_s

    def weighted_sum(self,h, h_s, r):
        """
        根据点积注意力权重对h和h_s进行加权求和
        参数:
        h (numpy array): 向量h
        h_s (numpy array): 向量h_s
        r (numpy array): 向量r

        返回:
        numpy array: 加权求和后的最终向量
        """
        weight_h, weight_h_s = self.compute_attention_weights(h, h_s, r)

        final_vector = weight_h.unsqueeze(1) * h + weight_h_s.unsqueeze(1) * h_s

        return final_vector


    def SPD_attention(self, entity_left, entity_right, rel_emb_forward, rel_emb_backward,
                      rel_embeds_left, rel_embeds_right, ent_embeds_left, ent_embeds_right):
        """
        ent： (few/b, dim)
        nn:  (few/b, max, 2*dim)
        output:  ()
        """
        # V_head = torch.relu(self.gate_w(torch.cat([rel_embeds_left, ent_embeds_left], dim=-1))) #拼接，线性变换，relu。h的邻域：rel_embeds_left，ent_embeds_left
        # V_tail = torch.relu(self.gate_w(torch.cat([rel_embeds_right, ent_embeds_right], dim=-1))) #t的邻域


        g = torch.sigmoid(rel_embeds_left@ self.W_g + ent_embeds_left @ self.U_g + self.b_g)

        V_head = torch.relu(g * rel_embeds_left + (1 - g) * ent_embeds_left)

        gr = torch.sigmoid(rel_embeds_right@ self.W_r + ent_embeds_right @ self.U_r + self.b_r)
        V_tail = torch.relu(gr * rel_embeds_right + (1 - gr) * ent_embeds_right)

        # learning h's task-aware representation
        head_nn_rel_aware = self.task_aware_attention_module(q=rel_emb_forward, k=rel_embeds_left, v=V_head)  # (few/b, dim)
        #方法一：
        head_nn_rel_aware = torch.relu(self.Linear_tail(head_nn_rel_aware) + self.Linear_head(entity_left)) #公式（7）
        enhanced_head_ = self.layer_norm(head_nn_rel_aware + entity_left)

        #方法二：
        # enhanced_head_ = self.layer_norm(torch.relu(self.weighted_sum(head_nn_rel_aware, entity_left, rel_emb_forward)))

        #方法三：
        # g1 = torch.sigmoid(self.W_g1(torch.cat([head_nn_rel_aware,entity_left],dim=-1)))  # 公式（7）
        # enhanced_head_ = self.layer_norm(g1*head_nn_rel_aware + (1-g1)*entity_left)



        # learning t's task-aware representation
        tail_nn_rel_aware = self.task_aware_attention_module(q=rel_emb_backward, k=rel_embeds_right, v=V_tail)  # (few/b, dim)
        tail_nn_rel_aware = torch.relu(self.Linear_tail(tail_nn_rel_aware) + self.Linear_head(entity_right)) #公式（7）加入门控
        enhanced_tail_ = self.layer_norm(tail_nn_rel_aware + entity_right)

        # enhanced_tail_ = self.layer_norm(torch.relu(self.weighted_sum(tail_nn_rel_aware, entity_right, rel_emb_backward)))

        # g2 = torch.sigmoid(self.W_g2(torch.cat([tail_nn_rel_aware,entity_right],dim=-1)))  # 公式（7）
        # enhanced_tail_ = self.layer_norm(g2*tail_nn_rel_aware + (1-g1)*entity_right)

        # learning h's entity-pair-aware representation
        head_nn_ent_aware = self.entity_pair_attention_module(q=enhanced_tail_, k=ent_embeds_left, v=V_head)
        head_nn_ent_aware = torch.relu(self.Linear_tail(head_nn_ent_aware) + self.Linear_head(entity_left)) #公式（11）加入门控
        enhanced_head = self.layer_norm(head_nn_ent_aware + entity_left)   ##

        # enhanced_head = self.layer_norm(torch.relu(self.weighted_sum(head_nn_ent_aware, entity_left, rel_emb_forward)))

        # g3 = torch.relu(self.W_g3(torch.cat([head_nn_ent_aware,entity_left],dim=-1))) #公式（11）加入门控
        # enhanced_head = self.layer_norm(g3*head_nn_ent_aware + (1-g3)*entity_left)   ##



        # learning t's entity-pair-aware representation
        tail_nn_ent_aware = self.entity_pair_attention_module(q=enhanced_head_, k=ent_embeds_right, v=V_tail)
        tail_nn_ent_aware = torch.relu(self.Linear_tail(tail_nn_ent_aware) + self.Linear_head(entity_right)) #公式（11）加入门控
        enhanced_tail = self.layer_norm(tail_nn_ent_aware + entity_right)  ##

        # enhanced_tail = self.layer_norm(torch.relu(self.weighted_sum(tail_nn_ent_aware, entity_right, rel_emb_backward)))

        # g4 = torch.relu(self.W_g4(torch.cat([tail_nn_ent_aware,entity_right],dim=-1))) #公式（11）加入门控
        # enhanced_tail = self.layer_norm(g4*tail_nn_ent_aware + (1-g4)*entity_right)  ##

        # computing entity-pair representation
        enhanced_pair = torch.cat([enhanced_head, enhanced_tail], dim=-1)
        ent_pair_rep = self.bi_MLP(enhanced_pair)


        # g5 = torch.sigmoid(enhanced_head@ self.W_g5 + enhanced_tail @ self.U_g5 + self.b_g5)

        # ent_pair_rep = torch.relu(g5 * enhanced_head + (1 - g5) * enhanced_tail)

        return ent_pair_rep


    def forward(self, entity_pairs, entity_meta):

        entity = self.dropout(self.symbol_emb(entity_pairs))  # (few/b, 2, dim)
        entity_left, entity_right = torch.split(entity, 1, dim=1)  # (few/b, 1, dim)
        entity_left = entity_left.squeeze(1)    # (few/b, dim)
        entity_right = entity_right.squeeze(1)   # (few/b, dim)


        entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

        relations_left = entity_left_connections[:, :, 0].squeeze(-1)
        entities_left = entity_left_connections[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # (few/b, max, dim)
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))   # (few/b, max, dim)

        relations_right = entity_right_connections[:, :, 0].squeeze(-1)
        entities_right = entity_right_connections[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (few/b, max, dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right)) # (few/b, max, dim)

        rel_emb = self.rel_w(entity_left, entity_right)
        rel_emb_forward, rel_emb_backward = torch.split(rel_emb, entity_left.size(-1), dim=-1)  # (few/b, dim)

        ent_pair_rep = self.SPD_attention(entity_left, entity_right, rel_emb_forward, rel_emb_backward,
                                          rel_embeds_left, rel_embeds_right, ent_embeds_left, ent_embeds_right)


        return ent_pair_rep

