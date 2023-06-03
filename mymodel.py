
import torch,pdb
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from util import gumbel_softmax
from modules import  AVWDCRNN,LearnablePositionalEncoding
from argu import args

class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()
        
        self.output_dim = args.output_dim
        self.hidden_dim = args.rnn_units
        self.horizon = args.horizon
        self.mem_dim = args.mem_dim
        self.mem_num = args.mem_num
        self.rnn_units =args.rnn_units
        self.num_nodes = args.num_nodes

        self.encoder = AVWDCRNN( args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.memory = self.construct_memory()


    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)#20,64 # (M, d)
        #memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True) #64,64   # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)#207,20 # project memory to embedding 
        #memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)##207,20 # project memory to embedding
        
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    # def query_memory(self, h_t:torch.Tensor):
    #     #h_t -> torch.Size([64, 207, 64]) 
    #     query = torch.matmul(h_t, self.memory['Wq']) # (B, N, d)  # [64, 207, 64] cross [64, 64]  -> [64, 207, 64]
    #     att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1) # alpha: (B, N, M) #[64, 207, 64] cross [64, 20] -> [64, 207, 20]
    #     #value = torch.matmul(att_score, self.memory['Memory']) # (B, N, d) # [64, 207, 20] cross [20, 64] ->[64, 207, 64]
    #     _, ind = torch.topk(att_score, k=2, dim=-1)
    #     pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d memory shape->[20, 64], ind shape ->[64, 207, 2]    #[64, 207, 64]
    #     neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
    #     return  query, pos, neg #value,


    def scaled_laplacian(self,node_embeddings1, node_num ,is_eval=False):
        #import pdb; pdb.set_trace()
        learned_graph = torch.mm(node_embeddings1, node_embeddings1.T)
        norm = torch.norm(node_embeddings1, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.T)
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.
        # learned_graph = F.sigmoid(learned_graph)
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
       
        # make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)

        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().to(adj.device)
        adj.masked_fill_(mask, 0)
       
        # d ->  diagonal degree matrix
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        ## L -> graph Laplacian
        L = -W
        L[range(len(L)), range(len(L))] = d
        try:
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0

        tilde = (2 * L / lambda_max - torch.eye(n).to(adj.device))
        self.adj = adj
        self.tilde = tilde
        return adj, tilde

    def forward(self, feat,adj_inp,cheb_polynomials, L_tilde, eval=False):
        
        # if not poe: 
            # weights = OrderedDict(self.named_parameters())
            # subweights =OrderedDict()

            # for k in weights.keys():
            #     if k.startswith("learnable_pos_encode"):
            #         stripped_key = k.replace("learnable_pos_encode.","")
            #         subweights[stripped_key] = torch.ones_like( weights[k]  )
            
            # mm= feat
            
            # feat = self.learnable_pos_encode.functional_forward(feat,subweights)
            # print(torch.equal(mm, feat))
            # exit()
        
        # if poe:
        #     feat = self.learnable_pos_encode(feat)
            
        
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory']) 
        #node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])

        NODE_NUM = feat.size(2)
        if self.train:
            _, learned_tilde = self.scaled_laplacian(node_embeddings1,NODE_NUM, is_eval=False)
        else:
            _, learned_tilde = self.scaled_laplacian(node_embeddings1,NODE_NUM, is_eval=True)

        
        if args.cheby:
            num_nodes = adj_inp.shape[1]
        else:
            num_nodes = adj_inp.shape[0]
        #feat - >(B,T,N,C) /torch.Size([Batch, 12, 170, 1])
        #feat= torch.unsqueeze(feat, 3)
        init_state = self.encoder.init_hidden(feat.shape[0],num_nodes)
        output, _ = self.encoder(feat, init_state, self.memory['We1'], learned_tilde,cheb_polynomials, L_tilde) 
        output = output[:, -1:, :, :]   
        
        output = self.end_conv((output))   
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, num_nodes)
        output = output.permute(0, 1, 3, 2)   

        return output
    