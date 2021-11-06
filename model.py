'''
Descripttion: model
version: 1.0
Author: Assassin 567
Date: 2021-11-03 10:40:02
LastEditors: Hello KG
LastEditTime: 2021-11-06 11:14:26
'''

import os
import sys
import time
from typing import List
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch_scatter import scatter_max, scatter_sum, scatter_softmax,scatter_add, scatter_mean

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)


class TimeEncode(torch.nn.Module):
    '''
    Description: the embedding of each entity is composed of three parts: static embedding; time-specific embedding-1 ; time-specific embedding-2 
    function: the time-specific embedding for entities  (time-specific embedding-2)
    '''    

    def __init__(self, expand_dim, entity_specific=False, num_entities=None, num_timestamps=None, device='cpu'):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific
        self.base_num = 50
        self.num_entities = num_entities
        self.num_timestamps = num_timestamps

        if entity_specific:

            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(np.random.randn(num_timestamps, self.base_num)).float())
            self.phase = torch.nn.Parameter(torch.zeros(self.base_num, self.time_dim).float()) #
        else:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
            
        self.ent_time_compare_base = nn.Linear(2*self.time_dim, self.time_dim, bias=False)
        self.ent_time_compare_base.weight.data[:,:self.time_dim] = torch.eye(self.time_dim)
        self.ent_time_compare_base.weight.data[:,self.time_dim:] = -1*torch.eye(self.time_dim)
        self.ent_time_act = torch.nn.LeakyReLU() #negative_slope=0.2
        

    def forward(self, ts, ts_base, entities=None):
        if self.entity_specific:
            assert sum(ts[0]<0)+sum(ts[1]<0)+sum(ts[2]<0) == 0
            assert np.max(ts[2]-ts[0])==1 and np.max(ts[0]-ts[1])==1 
            harmonic = torch.mm(self.basis_freq[ts[0]],self.phase)  # W * base_vec    base_vec*256
            reg_time = torch.norm(torch.mm(self.basis_freq[ts[1]]-self.basis_freq[ts[0]],self.phase),p=2,dim=1)+ \
                torch.norm(torch.mm(self.basis_freq[ts[2]]-self.basis_freq[ts[0]],self.phase),p=2,dim=1)
                
            harmonic_base = torch.mm(self.basis_freq[ts_base[0]],self.phase)  # W * base_vec    base_vec*256
            reg_time_base = torch.norm(torch.mm(self.basis_freq[ts_base[1]]-self.basis_freq[ts_base[0]],self.phase),p=2,dim=1)+ \
                torch.norm(torch.mm(self.basis_freq[ts_base[2]]-self.basis_freq[ts_base[0]],self.phase),p=2,dim=1)
            harmonic = self.ent_time_act(self.ent_time_compare_base(torch.cat((harmonic,harmonic_base),dim=1)))
            reg_time = torch.cat([reg_time, reg_time_base], axis=0) 
        else:
            batch_size = ts.size(0)
            seq_len = ts.size(1)
            ts = torch.unsqueeze(ts, dim=2)
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [batch_size, 1, time_dim]
            map_ts += self.phase.view(1, 1, -1)
            harmonic = torch.cos(map_ts)
            reg_time = 0
        return harmonic, reg_time

class TimeEncode_ori(torch.nn.Module):
    '''
    This class implemented the Bochner's time embedding
    expand_dim: int, dimension of temporal entity embeddings
    enitity_specific: bool, whether use entith specific freuency and phase.
    num_entities: number of entities.
    
    function: the time-specific embedding for entities  (time-specific embedding-1)
    
    ref : xERTE: Explainable Subgraph Reasoning for Forecasting on Temporal Knowledge Graphs.
    https://github.com/TemporalKGTeam/xERTE
    '''

    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        """
        :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
        :param entity_specific: if use entity specific time embedding  
        :param num_entities: number of entities
        refer to Self-attention with Functional Time Representation Learning for more detail
        """
        super(TimeEncode_ori, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific

        if entity_specific:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_entities, 1))
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_entities, 1))
        else:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        '''
        :param ts: [batch_size, seq_len]
        :param entities: which entities do we extract their time embeddings.
        :return: [batch_size, seq_len, time_dim]
        '''
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = torch.unsqueeze(ts, dim=2)
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(
                dim=1)  # 
            map_ts += self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1)  #
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class G3(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        """[summary]
        bilinear mapping along last dimension of x and y:
        output = MLP_1(x)^T A MLP_2(y), where A is two-dimenion matrix

        Arguments:
            left_dims {[type]} -- input dims of MLP_1
            right_dims {[type]} -- input dims of MLP_2
            output_dims {[type]} -- [description]
        """
        super(G3, self).__init__()
        self.dim_out = dim_out
        self.query_proj = nn.Linear(dim_in, 1, bias=False)
        nn.init.normal_(self.query_proj.weight, mean=0, std=np.sqrt(2.0 / (dim_in)))
        self.key_proj = nn.Linear(dim_out, dim_out//3, bias=False)
        nn.init.normal_(self.key_proj.weight, mean=0, std=np.sqrt(2.0 / (dim_out)))

        self.act_leaky = nn.Sigmoid()

        self.layernorm = torch.nn.LayerNorm(normalized_shape=dim_in,elementwise_affine=True)

    def forward(self, inputs):
        obj_later = inputs[1]
        triple_loss = torch.norm(inputs[3] + inputs[2] - obj_later,p=1 ,dim=1) # 
        rel_loss = torch.norm(inputs[4] - inputs[2],p=1 ,dim=1)  #
        return rel_loss, triple_loss # 

class AttentionFlow(nn.Module):
    def __init__(self, n_dims_in, n_dims_out, ratio_update=0, node_score_aggregation='sum',
                 device='cpu'):
        """[summary]

        Arguments:
            n_dims -- int, dimension of entity and relation embedding
            n_dims_sm -- int, smaller than n_dims to reduce the compuation consumption of calculating attention score
            ratio_update -- new node representation = ratio*self+(1-ratio)\sum{aggregation of neighbors' representation}
        """
        super(AttentionFlow, self).__init__()
        self.transition_fn = G3(6 * n_dims_in, 3 * n_dims_in)
        self.linear_between_steps = nn.Linear(n_dims_in, n_dims_out, bias=True)
        torch.nn.init.xavier_normal_(self.linear_between_steps.weight)
        
        self.linear_between_steps_time = nn.Linear(n_dims_in, n_dims_out, bias=True)
        torch.nn.init.xavier_normal_(self.linear_between_steps_time.weight)
        self.act_between_steps = torch.nn.LeakyReLU(negative_slope=0.2)

        self.node_score_aggregation = node_score_aggregation
        self.ratio_update = torch.nn.Parameter(torch.tensor(ratio_update))   #

        self.query_src_ts_emb = None
        self.query_rel_emb = None

        self.device = device
        
        

        self.linear_src_update = nn.Linear(n_dims_in*5 + n_dims_in*3, n_dims_in, bias=True) #MLP
        torch.nn.init.xavier_normal_(self.linear_src_update.weight)
        self.act_src_update = torch.nn.LeakyReLU(negative_slope=0.02) #act

        self.linear_rel_update = nn.Linear(n_dims_in*5 + n_dims_in*3 , n_dims_in, bias=True)
        torch.nn.init.xavier_normal_(self.linear_src_update.weight)
        self.act_rel_update = torch.nn.LeakyReLU(negative_slope=0.02)
        
        self.linear_src_update_1 = nn.Linear(n_dims_in*4, 1, bias=False) #MLP
        torch.nn.init.xavier_normal_(self.linear_src_update_1.weight)
        self.linear_rel_update_1 = nn.Linear(n_dims_in*4, 1, bias=False)
        torch.nn.init.xavier_normal_(self.linear_rel_update_1.weight)
        self.act_sigmoid = nn.Sigmoid()

        self.linear_trans = nn.ModuleList(nn.Linear(n_dims_in*3, n_dims_in, bias=False) for i in range(3))
        for ss in range(3):
            torch.nn.init.xavier_normal_(self.linear_trans[ss].weight)

        
        self.time_weight = nn.Linear(n_dims_in*7, 1, bias=True)
        torch.nn.init.xavier_normal_(self.time_weight.weight)
        
        self.step_score_add = torch.nn.Parameter(torch.ones(3, 1).float())

        self.query_obj_answer = nn.Linear(n_dims_in*7, 1, bias=True)  # act_rel_update
        torch.nn.init.xavier_normal_(self.query_obj_answer.weight)
        
        self.layernorm_1 = torch.nn.LayerNorm(normalized_shape=n_dims_in*7, elementwise_affine=True)


    def set_query_emb(self, query_src_ts_emb, query_rel_emb):
        self.query_src_ts_emb, self.query_rel_emb = query_src_ts_emb, query_rel_emb

    def set_query_reg(self, reg_query):
        self.reg_query = reg_query

    def set_query_time(self, query_ts_emb_special):
        self.query_ts_emb_special = query_ts_emb_special

    def _topk_att_score(self, edges, logits, k: int, tc=None):
        """
        :param edges: numpy array, (eg_idx, vi, ti, vj, tj, rel, node_idx_i, node_idx_j), dtype np.int32
        :param logits: tensor, same length as edges, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        pruned_edges, numpy.array, (eg_idx, vi, ts)
        pruned_logits, tensor, same length as pruned_edges
        origin_indices
        """
        if tc:
            t_start = time.time()
        res_edges = []
        res_logits = []
        res_indices = []
        for eg_idx in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg_idx
            orig_indices = np.arange(len(edges))[mask]
            masked_edges = edges[mask]
            masked_edges_logits = logits[mask]
            if masked_edges.shape[0] <= k:  #
                res_edges.append(masked_edges)
                res_logits.append(masked_edges_logits)
                res_indices.append(orig_indices)
            else: #
                topk_edges_logits, indices = torch.topk(masked_edges_logits, k)
                res_indices.append(orig_indices[indices.cpu().numpy()])
                try:
                    res_edges.append(masked_edges[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_logits.append(topk_edges_logits)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_edges, axis=0), torch.cat(res_logits, dim=0), np.concatenate(res_indices, axis=0)

    def _cal_attention_score(self, edges, memorized_embedding, rel_emb, src_ts_emb_special_1):
        """
        calculating node attention from memorized embedding
        """
        
        hidden_vi_orig = memorized_embedding[0] #
        hidden_vj_orig = memorized_embedding[2] #

        return self.cal_attention_score(edges[:, 8], hidden_vi_orig, hidden_vj_orig, rel_emb, src_ts_emb_special_1) #

    def cal_attention_score(self, query_idx, hidden_vi, hidden_vj, rel_emb, src_ts_emb_special):
        """
        calculate attention score between two nodes of edges
        wraped as a separate method so that it can be used for calculating attention between a node and it's full
        neighborhood, attention is used to select important nodes from the neighborhood
        :param query_idx: indicating in subgraph for which query the edge lies.
        """

        query_src_ts_emb_repeat = torch.index_select(self.query_src_ts_emb, dim=0,
                                                     index=torch.from_numpy(query_idx).long().to(
                                                         self.device))  #
        query_rel_emb_repeat = torch.index_select(self.query_rel_emb, dim=0,
                                                  index=torch.from_numpy(query_idx).long().to(
                                                      self.device))#

        transition_logits,  loght_emb = self.transition_fn(
            (hidden_vi, hidden_vj, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat, \
                src_ts_emb_special[0], src_ts_emb_special[1], src_ts_emb_special[2])) #

        return transition_logits,  loght_emb

    def forward(self, step_score_add_all, visited_node_score, visited_nodes_reg, src_ts_emb_special_set,sample_loss,  selected_edges_l=None, visited_node_representation=None, rel_emb_l=None,
                max_edges=10, analysis=False, tc=None):
        """calculate attention score

        Arguments:
            node_attention {tensor, num_edges} -- src_attention of selected_edges, node_attention[i] is the attention score
            of (selected_edge[i, 1], selected_edge[i, 2]) in eg_idx==selected_edge[i, 0]

        Keyword Arguments:
            selected_edges {numpy.array, num_edges x 9} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj,query-id) (default: {None})
            contain selfloop
            memorized_embedding torch.Tensor,
        return:
            pruned_edges, orig_indices
            updated_memorized_embedding:
            updated_node_score: Tensor, shape: n_new_node
            :param attended_nodes: 
        """

        transition_logits,  loght_emb = self._cal_attention_score(selected_edges_l[-1], visited_node_representation, rel_emb_l[-1], src_ts_emb_special_set)#

        transition_logits = torch.mm(torch.stack((transition_logits , sample_loss , \
             loght_emb ), dim=-1), -1*torch.abs(step_score_add_all))  # torch.zeros_like(transition_logits)
        
        
        target_score = transition_logits.squeeze()
        pruned_edges, _, orig_indices = self._topk_att_score(selected_edges_l[-1], target_score,
                                                                               max_edges) #

        loght_emb = loght_emb.unsqueeze(1)


        return target_score, visited_nodes_reg, pruned_edges, orig_indices

    def _update_node_representation_along_edges_old(self, edges, memorized_embedding, transition_logits):
        num_nodes = len(memorized_embedding)
        # update representation of nodes with neighbors
        # 1. message passing and aggregation
        sparse_index_rep = torch.from_numpy(edges[:, [6, 7]]).to(torch.int64).to(self.device)
        sparse_value_rep = transition_logits
        trans_matrix_sparse_rep = torch.sparse.FloatTensor(sparse_index_rep.t(), sparse_value_rep,
                                                           torch.Size([num_nodes, num_nodes])).to(self.device)
        updated_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_rep, memorized_embedding)
        # 2. linear
        updated_memorized_embedding = self.act_between_steps(self.linear_between_steps(updated_memorized_embedding))
        # 3. pass representation of nodes without neighbors, i.e. not updated
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, 6])).unsqueeze(
            1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)
        trans_matrix_sparse_identical = torch.sparse.FloatTensor(sparse_index_identical.t(), sparse_value_identical,
                                                                 torch.Size([num_nodes, num_nodes])).to(self.device)
        identical_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_identical, memorized_embedding)
        updated_memorized_embedding = updated_memorized_embedding + identical_memorized_embedding
        return updated_memorized_embedding

    def _update_node_representation_along_edges(self, edges, node_representation, transition_logits, linear_act=True):
        """
        :param edges:
        :param memorized_embedding:
        :param transition_logits:
        :param linear_act: whether apply linear and activation layer after message aggregation
        :return:
        """
        num_nodes = len(node_representation)
        sparse_index_rep = torch.from_numpy(edges[:, [6, 7]]).to(torch.int64).to(self.device)  #1307*2
        sparse_value_rep = (1 - self.ratio_update) * transition_logits #1307
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, 6])).unsqueeze(
            1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)
        sparse_index_self = torch.from_numpy(np.unique(edges[:, 6])).unsqueeze(1).repeat(1, 2).to(self.device)
        sparse_value_self = self.ratio_update * torch.ones(len(sparse_index_self)).to(self.device)
        sparse_index = torch.cat([sparse_index_rep, sparse_index_identical, sparse_index_self], axis=0)
        sparse_value = torch.cat([sparse_value_rep, sparse_value_identical, sparse_value_self])
        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index.t(), sparse_value,
                                                       torch.Size([num_nodes, num_nodes])).to(self.device)
        updated_node_representation = torch.sparse.mm(trans_matrix_sparse, node_representation)
        if linear_act:
            updated_node_representation = self.act_between_steps(self.linear_between_steps(updated_node_representation))

        return updated_node_representation

    def _update_node_representation_along_edges_new(self, edges, node_representation, pruned_rel_emb, transition_logits, linear_act=True):
        """
        :param edges:
        :param memorized_embedding:
        :param transition_logits:
        :param linear_act: whether apply linear and activation layer after message aggregation
        :return:
        """
        num_nodes = len(node_representation)
        input_rel = pruned_rel_emb# rel
        input_ent_1 = node_representation[edges[:, 6],:] #  src
        input_ent_2 = node_representation[edges[:, 7],:] # obj
        out_ent = [[] for i in range(3)]
        for i in range(3):
            out_ent[i] = self.linear_trans[i](torch.cat([input_rel, input_ent_1, input_ent_2], axis=1))# 
            out_ent[i] = self.act_between_steps(out_ent[i])  # LeakyReLU
        out_ent = sum(out_ent)/3
        out_ent = out_ent*transition_logits.unsqueeze(1)  #
        sparse_index_rep = torch.from_numpy(edges[:, [6]]).to(torch.int64).to(self.device) 
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, 6])).unsqueeze(1).to(self.device) #
        
        sparse_index = torch.cat([sparse_index_rep, sparse_index_identical], axis=0)  #
        spply_ent = node_representation[sparse_index_identical[:,0],:] #
        sparse_value = torch.cat([out_ent, spply_ent])  #
        updated_node_representation = scatter_add(sparse_value, sparse_index.squeeze(), dim=0)  #

        updated_node_representation = (1 - self.ratio_update)*updated_node_representation + self.ratio_update*node_representation #

        if linear_act:
            updated_node_representation = self.act_between_steps(self.linear_between_steps(updated_node_representation))

        return updated_node_representation


    def bypass_forward(self, embedding):
        return embedding

    def bypass_forward_time_encode(self, embedding):
        return embedding

    def query_src_update(self, new_query, rel_pass, nodes_src, nodes_obj, \
        src_ts_emb_special, src_ts_emb_special_new, obj_ts_emb_special):
        '''
        function:update the src of each query for each path 
        # self.linear_src_update = nn.Linear(n_dims_in*5, n_dims_out, bias=True) #MLP
        # self.act_src_update = torch.nn.LeakyReLU(negative_slope=0.2) #act
        '''
        
        # query update
        old_query_src = torch.index_select(self.query_src_ts_emb, dim=0,index=torch.from_numpy(new_query).long().to(self.device))  #query-src
        old_query_rel = torch.index_select(self.query_rel_emb, dim=0,index=torch.from_numpy(new_query).long().to(self.device))  #query-rel
        query_src_ts_emb = old_query_src + rel_pass # 
        query_rel_emb = old_query_rel - rel_pass  #

        return self.bypass_forward(query_src_ts_emb),  self.bypass_forward(query_rel_emb)  #
        
    def query_rel_update(self, new_query, rel_pass, nodes_pass):
        
        old_query_src = torch.index_select(self.query_src_ts_emb, dim=0,index=torch.from_numpy(new_query[:,3]).long().to(self.device))  #src
        old_query_rel = torch.index_select(self.query_rel_emb, dim=0,index=torch.from_numpy(new_query[:,3]).long().to(self.device))  #rel
        query_cat_rel = torch.cat([old_query_src, old_query_rel, rel_pass, nodes_pass], axis=1)  #concat
        return self.linear_rel_update(query_cat_rel)


    def cal_query_time_weight(self, query_idx, triple_loss, answer_embed):
        '''
        function: 
        '''
        query_rel_emb_repeat = torch.index_select(self.query_rel_emb, dim=0,
                                                  index=torch.from_numpy(query_idx).long().to(
                                                      self.device))#找出问题剩余的关系(关系累计误差)

        query_src_emb_repeat = torch.index_select(self.query_src_ts_emb, dim=0,
                                                  index=torch.from_numpy(query_idx).long().to(
                                                      self.device))#找出问题的剩余的主语  （与实际求得的答案相减，为实体误差累计加上关系误差）

        return torch.mm(torch.stack((torch.norm(query_rel_emb_repeat,p=1 ,dim=1) , triple_loss, \
             torch.norm(query_src_emb_repeat-answer_embed,p=1 ,dim=1) ), dim=-1), -1*torch.abs(self.step_score_add))



class xERTE(torch.nn.Module):
    def __init__(self, ngh_finder, num_entity=None, num_rel=None, timestamps=None, ent_time_set=None, emb_dim: List[int] = None,
                 DP_num_edges=40, DP_steps=3,
                 emb_static_ratio=1, diac_embed=False,
                 node_score_aggregation='sum', ent_score_aggregation='sum', max_attended_edges=20, ratio_update=0,
                 device='cpu', analysis=False, use_time_embedding=True, loss_margin=0, **kwargs):
        """[summary]

        Arguments:
            ngh_finder {[type]} -- an instance of NeighborFinder, find neighbors of a node from temporal KG
            according to TGAN scheme

        Keyword Arguments:
            num_entity {[type]} -- [description] (default: {None})
            num_rel {[type]} -- [description] (default: {None})
            embed_dim {[type]} -- [dimension of ERTKG embedding] (default: {None})
            attn_mode {str} -- [currently only prod is supported] (default: {'prod'})
            use_time {str} -- [use time embedding] (default: {'time'})
            agg_method {str} -- [description] (default: {'attn'})
            tgan_num_layers {int} -- [description] (default: {2})
            tgan_n_head {int} -- [description] (default: {4})
            null_idx {int} -- [description] (default: {0})
            drop_out {float} -- [description] (default: {0.1})
            seq_len {[type]} -- [description] (default: {None})
            max_attended_nodes {int} -- [max number of nodes in attending-from horizon] (default: {20})
            ratio_update: new node representation = ratio*self+(1-ratio)\sum{aggregation of neighbors' representation}
            device {str} -- [description] (default: {'cpu'})
        """
        super(xERTE, self).__init__()
        
        self.smooth_label = 0.01 #0.1 #0.01 #0.001  #
        print('label smooth : ', self.smooth_label )
        
        self.step_score_add_all = torch.nn.Parameter(torch.ones(3, 1).float())

        self.self_triple_ent_transform = nn.Linear(3*emb_dim[0], emb_dim[0], bias=False)
        torch.nn.init.xavier_normal_(self.self_triple_ent_transform.weight)
        self.self_triple_ent_transform.to(device)
        self.self_act_relu = torch.nn.LeakyReLU() 
        
        emb_dim.append(emb_dim[-1])  #

        self.DP_num_edges = DP_num_edges #
        self.DP_steps = DP_steps #
        self.use_time_embedding = use_time_embedding
        self.ngh_finder = ngh_finder
        

        self.temporal_embed_dim = [int(emb_dim[_] * 2 / (1 + emb_static_ratio)) for _ in range(DP_steps)] #
        self.static_embed_dim = [emb_dim[_] * 2 - self.temporal_embed_dim[_] for _ in range(DP_steps)] #

        self.entity_raw_embed = torch.nn.Embedding(num_entity, self.static_embed_dim[0]).cpu() #
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = torch.nn.Embedding(num_rel + 1, emb_dim[0]).cpu() #
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = num_rel  # index of relation "selfloop"
        self.att_flow_list = nn.ModuleList([AttentionFlow(emb_dim[_], emb_dim[_ + 1], 
                                                          node_score_aggregation=node_score_aggregation,
                                                          ratio_update=ratio_update, device=device,)
                                            for _ in range(DP_steps+1)])
        if use_time_embedding:
            self.node_emb_proj = nn.Linear(2 * emb_dim[0], emb_dim[0])
            self.time_emb_comp = nn.Linear(2 * emb_dim[0], emb_dim[0])
            self.static_time_comb = nn.Linear(2 * emb_dim[0], emb_dim[0])
        else:
            self.node_emb_proj = nn.Linear(emb_dim[0], emb_dim[0])

        nn.init.xavier_normal_(self.node_emb_proj.weight)
        self.max_attended_edges = max_attended_edges
        

        self.timestamps_dic = np.zeros(max(timestamps)+1,dtype=np.int32)
        self.timestamps_dic[:] = -1
        for idx,ss in enumerate(timestamps):
           self.timestamps_dic[ss] =  idx
        

        self.timestamps_dic = np.zeros((num_entity,max(timestamps)+1),dtype=np.int32)
        self.timestamps_former_dic = np.zeros((num_entity,max(timestamps)+1),dtype=np.int32)
        self.timestamps_later_dic = np.zeros((num_entity,max(timestamps)+1),dtype=np.int32)
        self.timestamps_dic[:] = -1
        self.timestamps_former_dic[:] = -1
        self.timestamps_later_dic[:] = -1

        for idx,ss in enumerate(ent_time_set): #
            self.timestamps_dic[ss[0],ss[1]] =  idx
            if ss[0]==ent_time_set[idx-1][0]: #
                self.timestamps_former_dic[ss[0],ss[1]] =  idx-1
            else:
                self.timestamps_former_dic[ss[0],ss[1]] =  idx

            try:
                if ss[0]==ent_time_set[idx+1][0]: #
                    self.timestamps_later_dic[ss[0],ss[1]] =  idx+1
                else:
                    self.timestamps_later_dic[ss[0],ss[1]] =  idx
            except:  #
                self.timestamps_later_dic[ss[0],ss[1]] =  idx

        x_label,y_label = np.where(self.timestamps_dic!=-1)
        x_former = -1
        for idx in range(len(x_label)):
            if x_label[idx]==x_former:#
                y_former = y_later
                if x_label[idx+1]==x_former: #
                    y_later = y_label[idx+1]
                else:
                    y_later = max(timestamps)+1 #
            else:  #
                x_former = x_label[idx]
                y_former = 0
                try:
                    if x_label[idx+1]==x_label[idx]: #
                        y_later = y_label[idx+1]
                    else:   #
                        y_later = max(timestamps)+1
                except:
                    if idx+1==len(y_label):  #
                        y_later = max(timestamps)+1
            self.timestamps_dic[x_former, y_former:y_later] = self.timestamps_dic[x_label[idx], y_label[idx]]
            
        x_former = -1
        for idx in range(len(x_label)):
            if x_label[idx]==x_former: #
                y_former = y_later
                if x_label[idx+1]==x_former:
                    y_later = y_label[idx+1]+1
                else:
                    y_later = max(timestamps)+1
            else:  #
                x_former = x_label[idx]
                y_former = 0
                try:
                    if x_label[idx+1]==x_label[idx]:
                        y_later = y_label[idx+1]+1
                    else:
                        y_later = max(timestamps)+1
                except:
                    if idx+1==len(y_label):
                        y_later = max(timestamps)+1
            self.timestamps_former_dic[x_former, y_former:y_later] = self.timestamps_dic[x_label[idx], y_label[idx]]

        x_former = -1
        for idx in range(len(x_label)):
            if x_label[idx]==x_former: #
                y_former = y_later
                if x_label[idx+1]==x_label[idx]:
                    y_later = y_label[idx]
                else:
                    y_later = max(timestamps)+1
            else:  #
                x_former = x_label[idx]
                y_former = 0
                try:
                    if x_label[idx+1]==x_label[idx]:
                        y_later = y_label[idx]
                    else:
                        y_later = max(timestamps)+1
                except:
                    if idx+1==len(y_label):
                        y_later = max(timestamps)+1
            self.timestamps_later_dic[x_former, y_former:y_later] = self.timestamps_dic[x_label[idx], y_label[idx]]
        
        if use_time_embedding:
            self.time_encoder = TimeEncode(expand_dim=self.temporal_embed_dim[0], entity_specific=diac_embed,
                                           num_entities=num_entity, num_timestamps= len(ent_time_set), device=device)  #
            
            self.time_encoder_ori = TimeEncode_ori(expand_dim=self.temporal_embed_dim[0], entity_specific=diac_embed,
                                           num_entities=num_entity, device=device)  #

        self.ent_spec_time_embed = diac_embed

        self.device = device
        self.analysis = analysis
        self.ent_score_aggregation = ent_score_aggregation

        self.res2query_node_score = nn.Linear(2, 1, bias=False)
        torch.nn.init.xavier_normal_(self.res2query_node_score.weight)
        self.act_res2query_node = torch.nn.LeakyReLU(negative_slope=0.02)

        self.param = Parameter(torch.Tensor((0.3,)))  #
        self.loss_margin = loss_margin

    def set_init(self, src_idx_l, rel_idx_l, cut_time_l, train_flag, target_idx_l):
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.cut_time_l = cut_time_l
        self.sampled_edges_l = []
        self.rel_emb_l = []
        self.node2index = {(i, src, ts): i for i, (src, rel, ts) in
                           enumerate(zip(src_idx_l, rel_idx_l, cut_time_l))}  # (eg_idx, ent, ts) -> node_idx
        self.num_existing_nodes = len(src_idx_l)

        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device)  #static-query-src
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)  # query-rel
        if self.use_time_embedding: #
            if self.ent_spec_time_embed:
                query_ts_emb_ori = self.time_encoder_ori(
                    torch.zeros(len(self.cut_time_l), 1).to(torch.float32).to(self.device),
                    entities=self.src_idx_l)  

                cut_time_l_in = [self.timestamps_dic[self.src_idx_l,self.cut_time_l], \
                    self.timestamps_former_dic[self.src_idx_l,self.cut_time_l], \
                        self.timestamps_later_dic[self.src_idx_l,self.cut_time_l]]
                query_ts_emb_special, reg_query = self.time_encoder(cut_time_l_in, cut_time_l_in)  

                query_ts_emb = query_ts_emb_ori.squeeze()  #
            else:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).to(torch.float32).to(self.device))
            query_ts_emb = torch.squeeze(query_ts_emb, 1) #

            # new
            query_src_ts_emb = self.self_triple_ent_transform(torch.cat((query_src_emb, query_ts_emb, query_ts_emb_special,), dim=-1))#
            query_src_ts_emb = self.self_act_relu(query_src_ts_emb)  #
            
            self.query_time_emb = query_ts_emb
        else:
            query_src_ts_emb = self.node_emb_proj(query_src_emb)

        self.att_flow_list[0].set_query_emb(query_src_ts_emb, query_rel_emb)  #
        if train_flag:
            cut_time_l_in = [self.timestamps_dic[target_idx_l,self.cut_time_l], \
                self.timestamps_former_dic[target_idx_l,self.cut_time_l], \
                    self.timestamps_later_dic[target_idx_l,self.cut_time_l]]
            _, reg_query_target = self.time_encoder(cut_time_l_in,cut_time_l_in)  #
            reg_query = torch.cat((reg_query, reg_query_target)) #
            self.att_flow_list[0].set_query_reg(reg_query) 
        else:
            self.att_flow_list[0].set_query_reg(reg_query)  #
    
        
        self.att_flow_list[0].set_query_time(query_ts_emb_special)
        
        return query_ts_emb_special

    def initialize(self):
        """get initial node (entity+time) embedding and initial node score
        Returns:
            attending_nodes, np.array -- n_attending_nodes x 3, (eg_idx, entity_id, ts)
            attending_node_attention, np,array -- n_attending_nodes, (1,)
            memorized_embedding, dict ((entity_id, ts): TGAN_embedding)
        """
        eg_idx_l = np.arange(len(self.src_idx_l), dtype=np.int32)
        att_score = np.ones_like(self.src_idx_l, dtype=np.float32) * (1 - 1e-8)

        attended_nodes = np.stack([eg_idx_l, self.src_idx_l, self.cut_time_l, np.arange(len(self.src_idx_l)), np.arange(len(eg_idx_l))], axis=1) #
        visited_nodes_score = torch.from_numpy(att_score).to(self.device) #
        visited_nodes = attended_nodes[:,:4] #


        visited_node_representation = self.att_flow_list[0].query_src_ts_emb #
        visited_nodes_reg =  self.att_flow_list[0].reg_query  #
        return attended_nodes, visited_nodes, visited_nodes_score, visited_node_representation, visited_nodes_reg

    def forward(self, sample, train_flag=False, p2o=None):
        src_idx_l, rel_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.ts  #
        if train_flag: #
            self.target_loss = self.get_target_loss(src_idx_l, rel_idx_l, cut_time_l,sample.target_idx) #
        else:
            self.target_loss = self.get_target_loss(src_idx_l, rel_idx_l, cut_time_l,[])

        p2o_set = None
        query_idx_l = sample.rel_idx  #
        if p2o!=None:
            p2o_set = [] #
            
            for  s in rel_idx_l:
                p2o_set.append(list(set(p2o[s])))
        
        src_ts_emb_special = self.set_init(src_idx_l, rel_idx_l, cut_time_l, train_flag, sample.target_idx)  #
        attended_nodes, visited_nodes, visited_node_score, visited_node_representation, visited_nodes_reg = self.initialize()
        for step in range(self.DP_steps): #
            attended_nodes, visited_nodes, no_pruned_score, no_pruned_edge_l, orig_indices, visited_nodes_reg, src_ts_emb_special, \
                query_src_new_emb, query_rel_new_emb, loss_error = \
                self._flow(attended_nodes, visited_nodes, visited_node_score, visited_nodes_reg, src_ts_emb_special, step, p2o_set, \
                    query_idx_l=query_idx_l,cut_time_l=cut_time_l)  #

            self.att_flow_list[step+1].set_query_emb(query_src_new_emb, query_rel_new_emb)  #


            query_rel_emb_repeat_pruned = no_pruned_score.unsqueeze(1)  #采样后的路径的打分（没有经过裁剪）
            error_loss = loss_error


            if step==0:
                attended_nodes_all = no_pruned_edge_l[:,[0, 3, 4, 7,8]] # attended_nodes 节点
                error_loss_all = error_loss
                query_rel_emb_repeat_all = query_rel_emb_repeat_pruned  #得分
            else:
                attended_nodes_all = np.concatenate((attended_nodes_all, no_pruned_edge_l[:,[0, 3, 4, 7,8]]),axis=0) 
                query_rel_emb_repeat_all = torch.cat((query_rel_emb_repeat_all, query_rel_emb_repeat_pruned),dim=0)
                if train_flag:
                    error_loss_all = torch.cat((error_loss_all, error_loss),dim=0) # error_loss


            torch.cuda.empty_cache()

        logits_rel,que_ent_idx = scatter_max(query_rel_emb_repeat_all.squeeze(), torch.LongTensor(attended_nodes_all[:, 3]).to(self.device)) 


        que_ent_idx_edx = torch.where(que_ent_idx < len(attended_nodes_all))[0]  #找出存在值的索引
        logits_rel = logits_rel[que_ent_idx_edx]  #找出存在的得分
        attended_nodes_all = attended_nodes_all[np.array(que_ent_idx[que_ent_idx_edx].cpu()),:4] #找出的index
        
        logits_rel_idx = attended_nodes_all[:, 0] # np.concatenate((attended_nodes_all[:, 0],np.array([128])))[que_ent_idx.cpu()]  #记录对应的query idx（设定一个不存在的值）
        logits_rel_idx_tensor = torch.LongTensor(logits_rel_idx).to(self.device)
        logits_rel_mean  = scatter_mean(logits_rel, logits_rel_idx_tensor) #(按照query索引）求各个batch的均值
        logits_rel_max, _  = scatter_max(logits_rel, logits_rel_idx_tensor) #求各个batch的最大值
        # logits_rel_num  = scatter_sum(torch.ones_like(logits_rel), torch.LongTensor(logits_rel_idx).to(self.device)) 
        variance = torch.pow(logits_rel-logits_rel_mean[logits_rel_idx],2)  #误差的平方
        variance = scatter_mean(variance, logits_rel_idx_tensor) #误差平方的均值 => 作为方差
        logits_rel = (logits_rel-logits_rel_max[logits_rel_idx])/ torch.pow(variance[logits_rel_idx]+1e-30,0.5)  #减去最大值 / 方差
        

        logits_rel = scatter_softmax(logits_rel, logits_rel_idx_tensor) #softmax，


        torch.cuda.empty_cache()

        entity_att_score, entities = self.get_entity_attn_score(logits_rel, attended_nodes_all)  #
        if train_flag:
            return entity_att_score, entities , torch.mean(visited_nodes_reg) ,  torch.mean(error_loss_all)# train
        else:
            return entity_att_score, entities , torch.mean(visited_nodes_reg) # train


    def _flow(self, attended_nodes, visited_nodes, visited_node_score, visited_nodes_reg, src_ts_emb_special, \
        step, p2o, tc=None, query_idx_l=None, cut_time_l=None):
        """[summary]
        
        Arguments:
            visited_nodes {numpy.array} -- num_nodes_visited x 4 (eg_idx, entity_id, ts, node_idx), dtype: numpy.int32, sort (eg_idx, ts, entity_id)
            all nodes visited during the expansion
            visited_node_score {Tensor} -- num_nodes_visited, dtype: torch.float32
            visited_node_representation {Tensor} -- num_nodes_visited x emb_dim_l[step]
            visited_node_score[node_idx] is the prediction score of node_idx
            visited_node_representation[node_idx] is the hidden representation of node_idx
        return:
            pruned_node {numpy.array} -- num_nodes_ x 4 (eg_idx, entity_id, ts, node_idx) sorted by (eg_idx, ts, entity_id)
            new_node_score {Tensor} -- new num_nodes_visited
            so that new_node_score[i] is the node prediction score of??
            updated_visited_node_representation: Tensor -- num_nodes_visited x emb_dim_l[step+1]
        """

        sampled_edges, new_sampled_nodes = self._get_sampled_edges(attended_nodes, cut_time_l,
                                                                    num_neighbors=self.DP_num_edges[step],
                                                                    step=step,
                                                                    add_self_loop=True, tc=tc,p2o=p2o,query_idx_l=query_idx_l)#

        self.sampled_edges_l.append(sampled_edges)  #

        sample_loss, loss_error, sample_embedding = self.get_query_answer_loss(self.target_loss, sampled_edges[:,3], sampled_edges[:,4], sampled_edges[:,0])


        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)  #
        self.rel_emb_l.append(rel_emb) #

        src_ts_emb_special_former = src_ts_emb_special[sampled_edges[:, 9]]
        src_ts_emb_special_new, obj_ts_emb_special , ent_time_reg = self.get_time_emb(sampled_edges[:,1], sampled_edges[:,4], sampled_edges[:,3], \
            self.target_loss[3][sampled_edges[:, 0]]) #

        src_ts_emb_special_set = [src_ts_emb_special_former, src_ts_emb_special_new, obj_ts_emb_special , ent_time_reg]

        visited_node_representation = self.get_visited_node_representation(self.sampled_edges_l[-1], cut_time_l[self.sampled_edges_l[-1][:,0]], src_ts_emb_special_set)
        
        # pruned
        no_pruned_score,visited_nodes_reg, pruned_edges, orig_indices = \
            self.att_flow_list[step](self.step_score_add_all, visited_node_score, visited_nodes_reg, src_ts_emb_special_set, sample_loss,
                                     selected_edges_l=self.sampled_edges_l,
                                     visited_node_representation=visited_node_representation,
                                     rel_emb_l=self.rel_emb_l,
                                     max_edges=self.max_attended_edges, tc=tc)#

        assert len(pruned_edges) == len(orig_indices)  #


        no_pruned_edge_l = sampled_edges
        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]

        # 
        src_ts_emb_special_former = src_ts_emb_special_former[orig_indices]
        src_ts_emb_special_new = src_ts_emb_special_new[orig_indices]
        obj_ts_emb_special = obj_ts_emb_special[orig_indices]
        # ent_time_reg = ent_time_reg[orig_indices]
        
        visited_nodes_reg = torch.cat([visited_nodes_reg, ent_time_reg], axis=0) 

        query_index = np.arange(len(pruned_edges)) #

        updated_attended_nodes =  np.concatenate([pruned_edges[:, [0, 3, 4, 7]], query_index.reshape(-1,1)], axis=1)  #

        rel_pass = self.rel_emb_l[-1]  #
        nodes_src = visited_node_representation[0][orig_indices] #
        nodes_obj = visited_node_representation[2][orig_indices] #

        query_src_new_emb, query_rel_new_emb \
            = self.att_flow_list[step].query_src_update(pruned_edges[:, 8], rel_pass, nodes_src, nodes_obj, \
                src_ts_emb_special_former, src_ts_emb_special_new, obj_ts_emb_special)   #

        return updated_attended_nodes, visited_nodes, no_pruned_score, no_pruned_edge_l, orig_indices,  visited_nodes_reg, obj_ts_emb_special, \
            query_src_new_emb, query_rel_new_emb, loss_error

    def loss(self, entity_att_score, entities, target_idx_l, batch_size, gradient_iters_per_update=1, loss_fn='BCE'):
        one_hot_label = torch.from_numpy(
            np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(self.device)  #

        label_average = 1/scatter_sum(torch.ones_like(one_hot_label), torch.LongTensor(entities[:,0]).to(self.device))[entities[:,0]]
        try:
            assert gradient_iters_per_update > 0

            one_hot_label = one_hot_label*(1-self.smooth_label) + self.smooth_label*label_average  #inv Label smooth
            if loss_fn == 'BCE':
                if gradient_iters_per_update == 1:
                    loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
                else:
                    loss = torch.nn.BCELoss(reduction='sum')(entity_att_score, one_hot_label)
                    loss /= gradient_iters_per_update * batch_size
            else:
                # CE has problems
                if gradient_iters_per_update == 1:
                    loss = torch.nn.NLLLoss()(entity_att_score, one_hot_label)
                else:
                    loss = torch.nn.NLLLoss(reduction='sum')(entity_att_score, one_hot_label)
                    loss /= gradient_iters_per_update * batch_size
        except:
            print(entity_att_score)
            entity_att_score_np = entity_att_score.cpu().detach().numpy()
            print("all entity score smaller than 1:", all(entity_att_score_np < 1))
            print("all entity score greater than 0:", all(entity_att_score_np > 0))
            raise ValueError("Check if entity score in (0,1)")
        return loss


    def get_time_emb(self, src_idx_l, cut_time_l, obj_idx_l, query_base_time):

        query_base_time_in = [self.timestamps_dic[src_idx_l,query_base_time], \
            self.timestamps_former_dic[src_idx_l,query_base_time], \
                self.timestamps_later_dic[src_idx_l,query_base_time] ]

        cut_time_l_in = [self.timestamps_dic[src_idx_l,cut_time_l], \
            self.timestamps_former_dic[src_idx_l,cut_time_l], \
                self.timestamps_later_dic[src_idx_l,cut_time_l] ]

        src_hidden_time, src_reg = self.time_encoder(cut_time_l_in, query_base_time_in) #

        cut_time_l_in = [self.timestamps_dic[obj_idx_l,cut_time_l], \
            self.timestamps_former_dic[obj_idx_l,cut_time_l], \
                self.timestamps_later_dic[obj_idx_l,cut_time_l] ]

        query_base_time_in = [self.timestamps_dic[obj_idx_l,query_base_time], \
            self.timestamps_former_dic[obj_idx_l,query_base_time], \
                self.timestamps_later_dic[obj_idx_l,query_base_time] ]

        obj_hidden_time, obj_reg = self.time_encoder(cut_time_l_in, query_base_time_in) #

        return src_hidden_time ,obj_hidden_time, torch.cat([src_reg, obj_reg],dim=-1)  #


    def get_time_emb_2(self,obj_idx_l, obj_time, base_time):
        
        cut_time_l_in = [self.timestamps_dic[obj_idx_l,obj_time], \
            self.timestamps_former_dic[obj_idx_l,obj_time], \
                self.timestamps_later_dic[obj_idx_l,obj_time] ]

        cut_time_l_base = [self.timestamps_dic[obj_idx_l,base_time], \
            self.timestamps_former_dic[obj_idx_l,base_time], \
                self.timestamps_later_dic[obj_idx_l,base_time] ]


        obj_hidden_time, obj_reg = self.time_encoder(cut_time_l_in,cut_time_l_base) #


        return obj_hidden_time  #

    def get_target_loss(self, src_idx, rel_idx, cut_time_l,target_idx):
        
        # 
        cut_time_l_ori = cut_time_l - cut_time_l #query时间误差
        src_time_ori = self.time_encoder_ori(torch.from_numpy(cut_time_l_ori[:, np.newaxis]).to(self.device),
                                        entities=src_idx).squeeze()
        src_node = self.get_ent_emb(src_idx, self.device) 
        
        
        src_hidden_time = self.get_time_emb_2(src_idx, cut_time_l, cut_time_l)  #target-time

        # rel
        rel_emb = self.get_rel_emb(rel_idx, self.device) 

        # new
        query_input_emb = self.self_triple_ent_transform(torch.cat((src_node, src_time_ori, src_hidden_time), dim=-1))
        query_input_emb = self.self_act_relu(query_input_emb)  #src的表征
        
        
        # answer
        if len(target_idx)>0:
            obj_time_ori = self.time_encoder_ori(torch.from_numpy(cut_time_l_ori[:, np.newaxis]).to(self.device),
                                            entities=target_idx).squeeze()
            obj_node = self.get_ent_emb(target_idx, self.device)
            
            obj_hidden_time = self.get_time_emb_2(target_idx, cut_time_l, cut_time_l)

            
            # new 
            answer_embed =  self.self_triple_ent_transform(torch.cat((obj_node, obj_time_ori, obj_hidden_time), dim=-1))
            answer_embed = self.self_act_relu(answer_embed)
            
            triple_loss = query_input_emb + rel_emb- answer_embed
                
                
            triple_loss = torch.norm(triple_loss,p=1 ,dim=1)
        else:
            triple_loss = []

        return triple_loss, query_input_emb, rel_emb, cut_time_l #, src_hidden_time_1 # torch.cat((answer_embed, answer_ts_emb, obj_hidden_time), dim=-1)


    def get_query_answer_loss(self, target_loss, obj_idx, obj_time, query_idx):

        cut_time_l_ori = obj_time - target_loss[3][query_idx]   #obj时间误差
        obj_time_ori = self.time_encoder_ori(torch.from_numpy(cut_time_l_ori[:, np.newaxis]).to(self.device),
                                        entities=obj_idx).squeeze()
        obj_node = self.get_ent_emb(obj_idx, self.device)
        
        
        # rel
        obj_hidden_time = self.get_time_emb_2(obj_idx, obj_time, target_loss[3][query_idx])

        # 5-2-12
        rel_emb = target_loss[1][query_idx]  #query rel
        query_input_emb = target_loss[2][query_idx] #query src
        
        # new
        answer_embed =  self.self_triple_ent_transform(torch.cat((obj_node, obj_time_ori, obj_hidden_time), dim=-1))
        answer_embed = self.self_act_relu(answer_embed)
        
        triple_loss = torch.norm(query_input_emb + rel_emb- answer_embed , p=1 ,dim=1)
            
        if len(target_loss[0])>0:
            loss_error = self.loss_margin + target_loss[0][query_idx] - triple_loss
            loss_error = torch.clamp(loss_error, min=0)
        else:
            loss_error = []

        return triple_loss, loss_error, answer_embed # torch.cat((answer_embed, answer_ts_emb, obj_hidden_time), dim=-1)



    def get_visited_node_representation(self, sampled_edges_l, cut_time_l, src_ts_emb_special_set):
        '''
        function: calculate the embedding of entity for the path 
        param {*}
        return {*}
        '''
        cut_time_l_former = sampled_edges_l[:,2] - cut_time_l #query
        
        cut_time_l_later = sampled_edges_l[:,4] - cut_time_l #obj
        src_idx = sampled_edges_l[:,1]
        obj_idx = sampled_edges_l[:,3]

        # src
        src_node = self.get_ent_emb(src_idx, self.device) 

        src_time_former = self.time_encoder_ori(torch.from_numpy(cut_time_l_former[:, np.newaxis]).to(self.device),
                                        entities=src_idx).squeeze()
        src_ts_emb_former = self.self_triple_ent_transform(torch.cat((src_node, src_time_former, src_ts_emb_special_set[0]), dim=-1))  
        src_ts_emb_former = self.self_act_relu(src_ts_emb_former)


        src_time_later = self.time_encoder_ori(torch.from_numpy(cut_time_l_later[:, np.newaxis]).to(self.device),
                                        entities=src_idx).squeeze()
        src_ts_emb_later = self.self_triple_ent_transform(torch.cat((src_node, src_time_later, src_ts_emb_special_set[1]), dim=-1))  # 
        src_ts_emb_later = self.self_act_relu(src_ts_emb_later)
        
        # obj
        obj_time_ori = self.time_encoder_ori(torch.from_numpy(cut_time_l_later[:, np.newaxis]).to(self.device),
                                        entities=obj_idx).squeeze()
        obj_node = self.get_ent_emb(obj_idx, self.device)
        answer_ts_emb = self.self_triple_ent_transform(torch.cat((obj_node, obj_time_ori, src_ts_emb_special_set[2]), dim=-1))  #
        answer_ts_emb = self.self_act_relu(answer_ts_emb)

        return [src_ts_emb_former, src_ts_emb_later, answer_ts_emb] # 


        

    def get_entity_attn_score(self, logits, nodes, tc=None,logits_rel=None):
        if tc:
            t_start = time.time()
        entity_attn_score, entities = self._aggregate_op_entity(logits, nodes, self.ent_score_aggregation,logits_rel)
        if tc:
            tc['model']['entity_attn'] = time.time() - t_start
        return entity_attn_score, entities

    def _aggregate_op_entity(self, logits, nodes, aggr='sum',logits_rel=None):
        """aggregate attention score of same entity, i.e. same (eg_idx, v)
        Arguments:
            logits {Tensor} -- attention score
            nodes {numpy.array} -- shape len(logits) x 3, (eg_idx, v, t), sorted by eg_idx, v, t
        return:
            entity_att_score {Tensor}: shape num_entity
            entities: numpy.array -- shape num_entity x 2, (eg_idx, v)
            att_score[i] if the attention score of entities[i]
        """
        device = logits.get_device()
        if device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        #~~~~~~~~~
        num_nodes = len(nodes)
        entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)  # entities_idx表示第一次出现的位置，赋值相应的标签.
        sparse_index = torch.LongTensor(np.stack([entities_idx, np.arange(num_nodes)])) #将实体点（与时间无关） 与 图中表示的标签对应（与时间有关）
        sparse_value = torch.ones(num_nodes, dtype=torch.float)
        if aggr == 'mean':
            c = Counter([(node[0], node[1]) for node in nodes[:, :2]])
            target_node_cnt = torch.tensor([c[(_[0], _[1])] for _ in nodes[:, :2]])
            sparse_value = torch.div(sparse_value, target_node_cnt)

        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                       torch.Size([len(entities), num_nodes])).to(device)
        entity_att_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1)))#

        return entity_att_score, entities

    def _get_sampled_edges(self, attended_nodes, base_time_l, num_neighbors: int = 20, step=None, add_self_loop=True, tc=None,p2o=None,query_idx_l=None):
        """[summary]
        按照每个src取15个相邻节点最多取15个节点，然后将其进行整合。
        sample neighbors for attended_nodes from all events happen before attended_nodes
        with strategy specified by ngh_finder, selfloop is added
        attended nodes: nodes in the current subgraph
        Arguments:
            attended_nodes {numpy.array} shape: num_attended_nodes x 4 (eg_idx, vi, ti, node_idx), dtype int32
            -- [nodes (with time) in attended from horizon, for detail refer to ERTKG paper]

        Returns:
            sampled_edges: {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None}), sorted ascending by eg_idx, ti, vi, tj, vj, rel dtype int32
            new_sampled_nodes: {Tensor} shape: new_sampled_nodes
        """
        if tc:
            t_start = time.time()
        batch_idx_l = attended_nodes[:, 0] 
        src_idx_l = attended_nodes[:, 1] #
        cut_time_l = attended_nodes[:, 2] #
        node_idx_l = attended_nodes[:, 3] #
        query_id_l = attended_nodes[:, 4] #

        flag = int(step==self.DP_steps-1)  # 
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l,
            cut_time_l, base_time_l[batch_idx_l],
            num_neighbors=num_neighbors,
            p2o=p2o,
            query_idx_l=query_idx_l,batch_idx_l=batch_idx_l,flag=flag)

        src_last_idx = np.zeros_like(src_ngh_node_batch)
        for inx in range(len(src_last_idx)):
            src_last_idx[inx,:] = inx

        if self.ngh_finder.sampling == -1:  # full neighborhood, select neighbors with largest attention score
            assert step is not None

            selected_src_ngh_node_batch = []
            selected_src_ngh_eidx_batch = []
            selected_src_ngh_t_batch = []
            with torch.no_grad():
                for i in range(len(src_ngh_eidx_batch)):
                    src_ngh_nodes = src_ngh_eidx_batch[i]
                    if sum(src_ngh_nodes != -1) > num_neighbors:

                        mask = (src_ngh_nodes != -1)
                        src_ngh_nodes = src_ngh_nodes[mask]
                        src_ngh_eidx = src_ngh_eidx_batch[i][mask]
                        src_ngh_t = src_ngh_t_batch[i][mask]
                        src_node_embed, src_reg = self.get_node_emb(np.array([src_idx_l[i]] * len(src_ngh_nodes)),
                                                           np.array([cut_time_l[i]] * len(src_ngh_nodes)),
                                                           np.array([attended_nodes[i, 0] * len(src_ngh_nodes)]))
                        ngh_node_embed, ngh_reg = self.get_node_emb(src_ngh_nodes, src_ngh_t,
                                                           np.array([attended_nodes[i, 0] * len(src_ngh_nodes)]))
                        rel_emb = self.get_rel_emb(src_ngh_eidx, self.device)

                        att_scores = self.att_flow_list[step].cal_attention_score(
                            np.ones(len(src_ngh_nodes)) * attended_nodes[i, 0], src_node_embed, ngh_node_embed, rel_emb)
                        _, indices = torch.topk(att_scores, num_neighbors)
                        indices = indices.cpu().numpy()
                        indices_sorted_by_timestamp = sorted(indices, key=lambda x: (
                            src_ngh_t[x], src_ngh_nodes[x], src_ngh_eidx[x]))
                        selected_src_ngh_node_batch.append(src_ngh_nodes[indices_sorted_by_timestamp])
                        selected_src_ngh_eidx_batch.append(src_ngh_eidx[indices_sorted_by_timestamp])
                        selected_src_ngh_t_batch.append(src_ngh_t[indices_sorted_by_timestamp])
                    else:
                        selected_src_ngh_node_batch.append(src_ngh_nodes[-num_neighbors:])
                        selected_src_ngh_eidx_batch.append(src_ngh_eidx_batch[i][-num_neighbors:])
                        selected_src_ngh_t_batch.append(src_ngh_t_batch[i][-num_neighbors:])
                src_ngh_node_batch = np.stack(selected_src_ngh_node_batch)
                src_ngh_eidx_batch = np.stack(selected_src_ngh_eidx_batch)
                src_ngh_t_batch = np.stack(selected_src_ngh_t_batch)

        # add selfloop 
        if add_self_loop:
            src_ngh_node_batch = np.concatenate([src_ngh_node_batch, src_idx_l[:, np.newaxis]], axis=1)  #
            src_ngh_eidx_batch = np.concatenate(
                [src_ngh_eidx_batch, np.array([[self.selfloop] for _ in range(len(attended_nodes))], dtype=np.int32)],
                axis=1)
            src_ngh_t_batch = np.concatenate([src_ngh_t_batch, cut_time_l[:, np.newaxis]], axis=1)  #
            src_last_idx = np.concatenate([src_last_idx, src_last_idx[:,:1]], axis=1)  #
        # removed padded neighbors, with node idx == rel idx == -1
        src_ngh_node_batch_flatten = src_ngh_node_batch.flatten()
        src_ngh_eidx_batch_flatten = src_ngh_eidx_batch.flatten()
        src_ngh_t_batch_faltten = src_ngh_t_batch.flatten()
        src_last_idx = src_last_idx.flatten()
        eg_idx = np.repeat(attended_nodes[:, 0], num_neighbors + int(add_self_loop)) #
        mask = src_ngh_node_batch_flatten != -1 #

        sampled_edges = np.stack([eg_idx,
                                  np.repeat(src_idx_l, num_neighbors + int(add_self_loop)),
                                  np.repeat(cut_time_l, num_neighbors + int(add_self_loop)), \
                                  src_ngh_node_batch_flatten, src_ngh_t_batch_faltten, \
                                  src_ngh_eidx_batch_flatten, \
                                  np.repeat(node_idx_l, num_neighbors + int(add_self_loop))], axis=1)[mask]  #

        target_nodes_index = [] #
        new_sampled_nodes = [] #
        for eg, tar_node, tar_ts in sampled_edges[:, [0, 3, 4]]:
            if (eg, tar_node, tar_ts) in self.node2index.keys():
                target_nodes_index.append(self.node2index[(eg, tar_node, tar_ts)])  #
            else:
                self.node2index[(eg, tar_node, tar_ts)] = self.num_existing_nodes  #
                target_nodes_index.append(self.num_existing_nodes)
                new_sampled_nodes.append([eg, tar_node, tar_ts, self.num_existing_nodes]) #
                self.num_existing_nodes += 1

        sampled_edges = np.concatenate([sampled_edges, np.array(target_nodes_index)[:, np.newaxis]], axis=1)  #
        sampled_edges = np.concatenate([sampled_edges, np.repeat(query_id_l, num_neighbors + int(add_self_loop)).reshape(-1,1)[mask]], axis=1)#
        
        sampled_edges = np.concatenate([sampled_edges, src_last_idx[mask][:, np.newaxis]], axis=1) 
        new_sampled_nodes = sorted(new_sampled_nodes, key=lambda x: x[-1])
        new_sampled_nodes = np.array(new_sampled_nodes)


        if tc:
            tc['graph']['sample'] += time.time() - t_start

        return sampled_edges, new_sampled_nodes #

    def _topk_att_score(self, attending_nodes, attending_node_attention, k: int, tc=None):
        """

        :param attending_nodes: numpy array, N_visited_nodes x 4 (eg_idx, vi, ts, node_idx), dtype np.int32
        :param attending_node_attention: tensor, N_all_visited_nodes, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        attended_nodes, numpy.array, (eg_idx, vi, ts)
        attended_node_attention, tensor, attention_score, same length as attended_nodes
        attended_node_emb, tensor, same length as attended_nodes
        """
        if tc:
            t_start = time.time()
        res_nodes = []
        res_att = []
        attending_node_attention = attending_node_attention[
            torch.from_numpy(attending_nodes[:, 3]).to(torch.int64).to(self.device)]
        for eg_idx in sorted(set(attending_nodes[:, 0])):
            mask = attending_nodes[:, 0] == eg_idx
            masked_nodes = attending_nodes[mask]
            masked_node_attention = attending_node_attention[mask]
            if masked_nodes.shape[0] <= k:
                res_nodes.append(masked_nodes)
                res_att.append(masked_node_attention)
            else:
                topk_node_attention, indices = torch.topk(masked_node_attention, k)
                try:
                    res_nodes.append(masked_nodes[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_att.append(topk_node_attention)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_nodes, axis=0), torch.cat(res_att, dim=0)

    def get_ent_emb(self, ent_idx_l, device):
        """
        help function to get node embedding
        self.entity_raw_embed[0] is the embedding for dummy node, i.e. node non-existing

        Arguments:
            node_idx_l {np.array} -- indices of nodes
        """
        embed_device = next(self.entity_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.entity_raw_embed(torch.from_numpy(ent_idx_l).long().to(embed_device)).to(device)

    def get_rel_emb(self, rel_idx_l, device):
        """
        help function to get relation embedding
        self.edge_raw_embed[0] is the embedding for dummy relation, i.e. relation non-existing
        Arguments:
            rel_idx_l {[type]} -- [description]
        """
        embed_device = next(self.relation_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.relation_raw_embed(torch.from_numpy(rel_idx_l).long().to(embed_device)).to(device)
