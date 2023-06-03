import torch,pdb,os,gc,types
import torch.nn as nn
import torch.nn.functional as F
from modules import  LearnablePositionalEncoding
from collections import OrderedDict
from argu import args
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from statistics import mean
import numpy as np
# class FinalModel(nn.Module):
#     def __init__(self, model):
#         super().__init__()

#         self.po_enc = LearnablePositionalEncoding(max_length=args.num_nodes, embedding_dim=args.horizon )
#         self.model = model

#     def forward(self, x,adj_inp,cheb_polynomials, L_tilde, eval=False):
        
#         x  = self.po_enc(x,adj_inp,cheb_polynomials, L_tilde, eval)
#         out = self.model(x,adj_inp,cheb_polynomials, L_tilde,eval)
#         return out


class MAMLnew(nn.Module):
    def __init__(self, model, inner_lr, meta_lr,scaler, loss, inner_steps, tasks_per_meta_batch ):
        super(MAMLnew, self).__init__()

        self.scaler = scaler
        #self.model = model
        self.loss = loss
        self.task_num = tasks_per_meta_batch
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        # self.net = nn.Sequential( LearnablePositionalEncoding(max_length=args.num_nodes, embedding_dim=args.horizon ),
        #                                model)
        self.net = model #FinalModel(model)
        self.meta_optim= torch.optim.Adam(params=self.net.parameters(), lr=args.lr_init,
         eps=1.0e-8,weight_decay=0, amsgrad=False)
        
        
    

        self.update_lr = inner_lr
        self.meta_lr = meta_lr

        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well 
        self.task_num = tasks_per_meta_batch 

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def compute_loss(self,model,  params,data,graph_supports, cheb_polynomials ,L_tilde,label):
        y = torch.func.functional_call(model, params,(data,graph_supports, cheb_polynomials ,L_tilde))
        label = self.scaler.inverse_transform(label) 

        return self.loss(y, label)
    
    def Merge(self, dict1, dict2):
        res = {**dict1, **dict2}
        return res


    def forward(self, x_spt, y_spt, x_qry, y_qry , graph_supports, cheb_polynomials ,L_tilde):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, _,_,_,_ = x_spt.size()
        querysz = x_qry.size(1)
        
        #self.net.learnable_pos_encode.embedding.weight.requires_grad =False
        # for name, param in self.named_parameters():
        #     print(name, param)
            
        #     print(name)
        # self.net.learnable_pos_encode.scale.requires_grad = False
        # self.net.learnable_pos_encode.positional_embedding.weight.requires_grad= False

            # if name== 'learnable_pos_encode.embedding.weight.require':
            #     param.requires_grad= False
            #     print("Done......")
        
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is tfhe loss on step i

        subweight = OrderedDict()
        for name, param in self.net.po_enc.named_parameters():
            subweight[name]= param
            
        

        for i in range(self.task_num):

            out = self.net.model(x_spt[i],graph_supports, cheb_polynomials ,L_tilde)
            loss = self.loss(out,y_spt[i])
            grad = torch.autograd.grad(loss,self.net.model.parameters()) #,create_graph =True) #,retain_graph =True)
            fast_weights= {k: p - self.update_lr * g for k, g, p in zip(self.net.model.state_dict().keys(), grad, self.net.model.parameters())}


            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                loss_q= self.compute_loss(self.net,  dict(self.net.named_parameters()), x_qry[i],graph_supports,
                                           cheb_polynomials ,L_tilde, y_qry[i])
                losses_q[0] += loss_q
            
            with torch.no_grad():
            # this is the loss and accuracy after the first update
                merged = self.Merge(subweight,fast_weights)
                new= list(dict(self.net.named_parameters()).keys())
                merged= dict(zip(new, list(merged.values())))


                loss_q = self.compute_loss(self.net, merged, x_qry[i] ,graph_supports, cheb_polynomials ,
                                           L_tilde, y_qry[i])
                losses_q[1] += loss_q
            
            for k in range(1, self.update_step):
                loss = self.compute_loss(self.net.model,  fast_weights ,x_spt[i] ,graph_supports, cheb_polynomials ,L_tilde, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights.values())
                fast_weights= {k: p - self.update_lr * g for k, g, p in zip(self.net.model.state_dict().keys(), grad,  fast_weights.values())}

                merged = self.Merge(subweight,fast_weights)
                merged= dict(zip(new, list(merged.values())))

                loss_q =self.compute_loss(self.net, merged ,x_qry[i] ,graph_supports, cheb_polynomials ,
                                          L_tilde, y_qry[i])

                losses_q[k + 1] += loss_q

                    # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        #return (loss_q.detach().item() , self.net)
        return  loss_q.detach().item()

    def finetunning(self, x_spt, y_spt, x_qry, y_qry,  graph_supports, cheb_polynomials ,L_tilde):

        params = deepcopy(self.net.state_dict())  
        
        param_list= []
        for k, v in params.items():
            v.requires_grad = True
            param_list.append(v)
        
        loss= self.compute_loss(self.net, params, x_qry,graph_supports, cheb_polynomials ,L_tilde, y_qry)
        grad = torch.autograd.grad(loss, param_list)
        fast_weights= {k: p - self.update_lr * g for k, g, p in zip(self.net.state_dict().keys(), grad, self.net.parameters())}

        with torch.no_grad():
            loss_q= self.compute_loss(self.net, params, x_qry,graph_supports, cheb_polynomials ,L_tilde, y_qry)

        with torch.no_grad():
        # this is the loss and accuracy after the first update
            loss_q = self.compute_loss(self.net, fast_weights, x_qry ,graph_supports, cheb_polynomials ,L_tilde, y_qry)

        for k in range(1, self.update_step_test):
            loss= self.compute_loss(self.net, fast_weights, x_spt ,graph_supports, cheb_polynomials ,L_tilde, y_spt)
            grad = torch.autograd.grad(loss, fast_weights.values())
            fast_weights= {k: p - self.update_lr * g for k, g, p in zip(self.net.state_dict().keys(), grad,  fast_weights.values())}
            loss_q = self.compute_loss(self.net, fast_weights, x_qry ,graph_supports, cheb_polynomials ,L_tilde,  y_qry)

        # self.meta_optim.zero_grad()
        # loss_q.backward()
        # self.meta_optim.step()

        del param_list



        return loss_q.detach().item()
    
