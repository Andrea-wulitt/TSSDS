from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn.functional as F
import torch.nn as nn
import torch
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.rsf_rel_emb = nn.Embedding(self.params.num_rels, self.params.rsf_dim, sparse=False)
        self.rsf_emb = nn.Embedding(self.params.aug_num_rels, self.params.rsf_dim)
        self.gnn_out_dim = self.params.num_gcn_layers * self.params.emb_dim
        self.dropout1 = nn.Dropout(self.params.rs_dropout)

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.gnn_out_dim + 2 * self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.gnn_out_dim + self.params.rel_emb_dim, 1)
        
        self.conve = ConvE(params, len(relation2id))


    def forward(self, graph_data, contrastive_data):
        (rsf_list, g, rel_labels), (head_con_pos, head_con_neg, tail_con_pos, tail_con_neg) = graph_data, contrastive_data

        
        head_rsf_list = rsf_list[:, 0]
        head_rsf_sum = torch.sum(head_rsf_list, dim=1).unsqueeze(1)
        head_rsf = torch.sum(head_rsf_list.unsqueeze(2) * self.rsf_emb.weight, dim=1) / head_rsf_sum
       
        tail_rsf_list = rsf_list[:, 1]
        tail_rsf_sum = torch.sum(tail_rsf_list, dim=1).unsqueeze(1)
        tail_rsf = torch.sum(tail_rsf_list.unsqueeze(2) * self.rsf_emb.weight, dim=1) / tail_rsf_sum
       
        rsf_output = self.dropout1(torch.sum(head_rsf * self.rsf_rel_emb(rel_labels) * tail_rsf, dim=1, keepdim=True))

        ##############################################################################################################
        potential_rel = self.dropout1(self.conve(head_rsf, tail_rsf))
        ##############################################################################################################


        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               potential_rel,
                               self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)

        head_con_pos = head_con_pos.squeeze()
        d0, d1, d2 = head_con_pos.shape
        head_con_pos_list = head_con_pos.view((d0 * d1, d2))
        head_con_pos_sum = torch.sum(head_con_pos_list, dim=1).unsqueeze(1)
        head_con_pos_rsf = torch.sum(head_con_pos_list.unsqueeze(2) * self.rsf_emb.weight, dim=1) / head_con_pos_sum
         
        head_con_neg = head_con_neg.squeeze()
         
        head_con_neg_list = head_con_neg.view((d0 * d1, d2))
        head_con_neg_sum = torch.sum(head_con_neg_list, dim=1).unsqueeze(1)
        head_con_neg_rsf = torch.sum(head_con_neg_list.unsqueeze(2) * self.rsf_emb.weight, dim=1) / head_con_neg_sum
         
        tail_con_pos = tail_con_pos.squeeze()
         
        tail_con_pos_list = tail_con_pos.view((d0 * d1, d2))
        tail_con_pos_sum = torch.sum(tail_con_pos_list, dim=1).unsqueeze(1)
        tail_con_pos_rsf = torch.sum(tail_con_pos_list.unsqueeze(2) * self.rsf_emb.weight, dim=1) / tail_con_pos_sum
         
        tail_con_neg = tail_con_neg.squeeze()
        
        tail_con_neg_list = tail_con_neg.view((d0 * d1, d2))
        tail_con_neg_sum = torch.sum(tail_con_neg_list, dim=1).unsqueeze(1)
        tail_con_neg_rsf = torch.sum(tail_con_neg_list.unsqueeze(2) * self.rsf_emb.weight, dim=1) / tail_con_neg_sum

         
        head_rsf = head_rsf.unsqueeze(1).repeat(1, d1, 1).view((d0 * d1, self.params.rsf_dim))
        tail_rsf = tail_rsf.unsqueeze(1).repeat(1, d1, 1).view((d0 * d1, self.params.rsf_dim))
         
        con_head_pos_dis = F.pairwise_distance(head_rsf, head_con_pos_rsf, p=2)
        
        con_head_neg_dis = F.pairwise_distance(head_rsf, head_con_neg_rsf, p=2)
        
        con_tail_pos_dis = F.pairwise_distance(tail_rsf, tail_con_pos_rsf, p=2)
        
        con_tail_neg_dis = F.pairwise_distance(tail_rsf, tail_con_neg_rsf, p=2)

        con_pos_dis = torch.cat((con_head_pos_dis, con_tail_pos_dis))
        con_neg_dis = torch.cat((con_head_neg_dis, con_tail_neg_dis))


        if self.params.remove_rsf:
            return output, con_pos_dis, con_neg_dis
        else:
            return  output + rsf_output, con_pos_dis, con_neg_dis


class ConvE(nn.Module):
    def __init__(self, params, num_relations):
        super(ConvE, self).__init__()
        self.relnum = params.relnum 
        self.entity_dim = params.emb_dim
        self.emb_2D_d1 = 10 #args.emb_2D_d1
        self.emb_2D_d2 = 20 #args.emb_2D_d2
        self.num_relations =num_relations
       
        self.fcs = nn.Linear(params.rel_emb_dim*2, params.rel_emb_dim)
        self.rel_emb = nn.Embedding(params.num_rels, params.rel_emb_dim*2, sparse=False)
       


    def forward(self, e1, e2):
        r = torch.tensor(list(range(self.num_relations))).to(device=e1.device)
        R = self.rel_emb(r) #[rel_num,32]

        stacked_inputs = torch.cat([e1, e2], 1)   
       
        X = torch.mm(stacked_inputs, R.transpose(1, 0)) #[batch_size, entity_num]
        S = torch.sigmoid(X)  

        Sx, idx = torch.topk(S, k=self.relnum, dim=1) # [batch_size, 1, 1] [batch_size, 1] 
        Sx = Sx.unsqueeze(-1)
        R1 = self.fcs(R)
        R1 = F.relu(R1)
        S = R1[idx] * Sx # [batch_size, 1, 200]
        x = torch.sum(Sx, dim=1, keepdim=True)
        S = S / x
        pred_kg = torch.sum(S, dim=1)
        return pred_kg  #[batch_size, entity_num]
    
  