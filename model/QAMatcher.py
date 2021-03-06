import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.BertModel import BertEmbedding


class BertQAMatcher(nn.Module):
    def __init__(self, bert_embed_dim,
                 sim_mode='cosine', num_bert_layer=4,
                 dropout=0.0, bert_model_path=None):
        super(BertQAMatcher, self).__init__()
        self.bert_embed_dim = bert_embed_dim
        self.dropout = dropout
        self.sim_mode = sim_mode
        self.bert = BertEmbedding(bert_model_path, num_bert_layer,
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=True)
        self.bert_norm = nn.LayerNorm(self.bert_embed_dim, eps=1e-6)

        # self.attn_fc = nn.Sequential(
        #     nn.Linear(self.bert_embed_dim, self.bert_embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.bert_embed_dim, 1, bias=False)
        # )

        if self.sim_mode != 'cosine':
            self.repr2vec = nn.Linear(self.bert_embed_dim, 1)
            nn.init.xavier_uniform_(self.repr2vec.weight)

    def kmax_pooling(self, x, k=1):
        # (B, N, D) -> (B, k, D)
        return F.adaptive_max_pool1d(x.transpose(1, 2).contiguous(), k).transpose(1, 2).contiguous()

    def kmax_pooling2(self, x, k=1, dim=1):
        # (B, N, D)
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]  # values, indices
        return x.gather(dim, index)

    def pooling(self, x):
        # avg_pool = F.avg_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        # max_pool = F.max_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        # return torch.cat((avg_pool, max_pool), -1)

        # pool = F.max_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        pool = F.avg_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        return pool

    def attn_pooling(self, x):
        attn_score = self.attn_fc(x)  # (B, L, 1)
        attn_w = F.softmax(attn_score, dim=1)
        out = torch.sum(attn_w * x, dim=1)  # (B, D)
        return out

    def bert_params(self):
        return self.bert.bert.parameters()

    def bert_named_params(self):
        return self.bert.bert.named_parameters()

    def non_bert_params(self):
        bert_param_names = []
        for name, param in self.bert.bert.named_parameters():
            if param.requires_grad:
                bert_param_names.append(id(param))

        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append(param)
        return other_params

    def get_repr(self, sent_bert_inp):
        '''
        :param sent_bert_inp: (bert_ids, segments, bert_masks)
        :return:
        '''
        sent_embed = self.bert(*sent_bert_inp)
        sent_repr = self.bert_norm(sent_embed)
        if self.training:
            sent_repr = F.dropout(sent_repr, p=self.dropout, training=self.training)
        sent_vec = self.pooling(sent_repr)  # (B, D)
        return sent_vec

    def get_cos_sim(self, query, doc):
        query_vec = self.get_repr(query)  # (B, D)
        doc_vec = self.get_repr(doc)  # (B, D)
        cos_sim = F.cosine_similarity(query_vec, doc_vec, dim=1, eps=1e-6)
        return cos_sim

    def forward(self, query, doc):
        '''
        :param query / doc: bert_ids, segments, bert_masks
        :return:
        '''
        query_vec = self.get_repr(query)  # (B, D)
        doc_vec = self.get_repr(doc)  # (B, D)

        if self.sim_mode == 'cosine':
            sim = F.cosine_similarity(query_vec, doc_vec, dim=1, eps=1e-6)  # (B, )
        elif self.sim_mode == 'multineg':
            # sim = self.multiple_negatives_ranking_loss(query_vec, doc_vec)
            sim = self.circle_loss(query_vec, doc_vec)
        else:
            # diff = torch.norm(query_vec - doc_vec, p=2, dim=1)  # square root
            # diff = torch.norm(query_vec - doc_vec, p=1, dim=1)   # abs
            diff = (query_vec - doc_vec) ** 2
            sim = self.repr2vec(diff).squeeze()

        return sim

    def multiple_negatives_ranking_loss(self, embeddings_a, embeddings_b, scale=20.0):
        """  batch size is larger !!
        :param embeddings_a: (B, D)
        :param embeddings_b: (B, D)
        :param scale: Output of similarity function is multiplied by scale value
        :return: The scalar loss
        """
        # norm_a = F.normalize(embeddings_a)
        # norm_b = F.normalize(embeddings_b)
        norm_a = embeddings_a / embeddings_a.norm(dim=1)[:, None]
        norm_b = embeddings_b / embeddings_b.norm(dim=1)[:, None]
        scores = torch.mm(norm_a, norm_b.transpose(0, 1).contiguous()) * scale  # (B, B)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)   # Example a[i] should match with b[i]
        return F.cross_entropy(scores, labels)

    def circle_loss(self, embeddings_a, embeddings_b, margin=0.45, gamma=32):
        """
        :param embeddings_a: (B, D)
        :param embeddings_b: (B, D)
        :return: The scalar loss
        """
        bs = embeddings_a.size(0)
        norm_a = embeddings_a / embeddings_a.norm(dim=1)[:, None]
        norm_b = embeddings_b / embeddings_b.norm(dim=1)[:, None]
        neg_cos_all = torch.matmul(norm_a, norm_b.transpose(0, 1).contiguous())   # B * B

        pos_cosine = torch.diag(neg_cos_all).reshape(-1, 1)
        neg_mask = ~torch.eye(bs).bool().to(neg_cos_all.device)
        neg_cosine = neg_cos_all.masked_select(neg_mask).reshape(bs, bs-1)
        # Perturbed Circle Loss
        neg_loss = torch.sum(torch.exp(gamma * ((neg_cosine + margin) * (neg_cosine - margin))), dim=1)
        pos_loss = torch.sum(torch.exp(-gamma * ((1. + margin - pos_cosine) * (pos_cosine - 1. + margin))), dim=1)
        circle_loss = torch.mean(torch.log(1. + neg_loss * pos_loss))
        return circle_loss
