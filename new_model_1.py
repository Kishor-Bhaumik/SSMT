
import torch,pdb,os,gc,types
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import scaled_Laplacian,cheb_polynomial
from modules import  WeeklyPositionalEncoding, DailyPositionalEncoding,LearnablePositionalEncoding
from mymodel import MetaModel
from argu import args
from maml import MAMLnew
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.utils import get_adjacency_matrix, scaled_Laplacian, cheb_polynomial
from lib.metrics import MAE_torch



###

## notice: hardsigmoid is replaced with sigmoid from the module.py
###

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss



    
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.SmoothL1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

# adj_mx = torch.rand(30,30) # np.random.rand(307,307)
# x = torch.rand(8, 12, 30)p

model = MetaModel()

class FinalModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.po_enc = LearnablePositionalEncoding(max_length=args.num_nodes, embedding_dim=args.horizon )
        self.model = model

    def forward(self, x,adj_inp,cheb_polynomials, L_tilde, eval=False):
        
        x  = self.po_enc(x,adj_inp,cheb_polynomials, L_tilde, eval)
        out = self.model(x,adj_inp,cheb_polynomials, L_tilde,eval)
        return out

model = FinalModel(model)
#model.load_state_dict(torch.load('39_best_model.pt'))


for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)


# print(M(x,graph_supports).shape)

def split_batch(imgs, targets):
    support_imgs,query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets\




if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.SmoothL1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

L = loss

# adj_mx = torch.rand(30,30) # np.random.rand(307,307)
# x = torch.rand(8, 12, 30)
device =args.device # "cuda:1" #"cuda:0"
id_filename =None
adj_mx, distance_mx = get_adjacency_matrix(args.adj_filename, args.num_nodes, id_filename)

if torch.is_tensor(adj_mx):
    adj_mx = adj_mx.cpu().detach().numpy()
L_tilde = scaled_Laplacian(adj_mx)

cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(L_tilde,args.cheb_k)]
graph_supports = torch.stack(cheb_polynomials, dim=0)  # [k, n, m]
if not args.cheby:

    graph_supports =torch.tensor(adj_mx).to(args.device) 


train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)


num_feature_train = next(iter(train_loader))[0].size(2)
d_model_train= next(iter(train_loader))[0].size(1)

####
####
args.dataset = 'PEMSD4'
args.adj_filename= 'data/PeMSD4/distance.csv'
args.num_nodes = 307
args.batch_size = int(args.batch_size /2)
train_loader_fine, val_loader_fine, test_loader_fine, scaler_fine = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)


num_feature_fine = next(iter(train_loader_fine))[0].size(2)
d_model_train_fine= next(iter(train_loader_fine))[0].size(1)




adj_mx, distance_mx = get_adjacency_matrix(args.adj_filename, args.num_nodes, id_filename)

if torch.is_tensor(adj_mx):
    adj_mx = adj_mx.cpu().detach().numpy()
L_tilde_fine = scaled_Laplacian(adj_mx)

cheb_polynomials_fine = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(L_tilde_fine,args.cheb_k)]
graph_supports_fine = torch.stack(cheb_polynomials_fine, dim=0)  # [k, n, m]

if not args.cheby:
    graph_supports_fine =torch.tensor(adj_mx).to(args.device) 




inner_lr = 0.01
meta_lr = 0.001
inner_steps=2
tasks_per_meta_batch=2

# data,_ = next(iter(train_loader))


model=MAMLnew( model, inner_lr, meta_lr,scaler, L, inner_steps, tasks_per_meta_batch)

model = model.to(args.device)



def remove(file):

    if os.path.isfile(file):
        os.remove(file)
    return


remove('train.pt')
remove('fine.pt')



def make_datas(data,label):

        num_features = data.size(1) #12
        seq_len = data.size(2) #170
        week_pos_enc = WeeklyPositionalEncoding(d_model=num_features, max_seq_len=seq_len)
        daily_pos_enc = DailyPositionalEncoding(d_model=num_features, max_seq_len=seq_len)
        
        #1
        spt_data, qu_data = torch.tensor_split(data,2)
        spt_lab, qu_lab = torch.tensor_split(label,2)
        
        #2
        task1_sp_data, task2_sp_data = torch.tensor_split(spt_data,2)
        task1_sp_label, task2_sp_label = torch.tensor_split(spt_lab,2)

        task1_qu_data, task2_qu_data = torch.tensor_split(qu_data,2)
        task1_qu_label, task2_qu_label = torch.tensor_split(qu_lab,2)
        #3
        task1_sp_data = daily_pos_enc( task1_sp_data)
        # task1_qu_data = daily_pos_enc(task1_qu_data)

        task2_sp_data = week_pos_enc(task2_sp_data)
        # task2_qu_data = week_pos_enc(task2_qu_data)
        #4
        sp_data= torch.stack([task1_sp_data, task2_sp_data])
        sp_label= torch.stack([task1_sp_label,task2_sp_label])

        qu_data = torch.stack([task1_qu_data,task2_qu_data ])
        qu_label = torch.stack([task1_qu_label,task2_qu_label ])

        return sp_data,sp_label, qu_data,qu_label


'''
#train upto few epochs , then fine tune with few epochs

for epoch in range(args.epochs):
    
    loss_train= []
    for step, (data,label) in enumerate(train_loader):

        sp_data,sp_label, qu_data,qu_label = make_datas(data,label)
        x_spt, y_spt, x_qry, y_qry = sp_data.to(args.device), sp_label.to(args.device), qu_data.to(args.device), qu_label.to(args.device)
        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
        loss = model(x_spt, y_spt, x_qry, y_qry,   graph_supports, cheb_polynomials ,L_tilde)
        loss_train.append(loss)
        break

    print("train loss epoch", epoch, np.mean(loss_train))

    if epoch % 10 == 0:
        torch.save(model.net.memory.We1, 'train.pt')

        if not os.path.isfile('fine.pt'):
            node_num = 307
            model.net.memory.We1 = nn.Parameter(torch.randn(node_num, args.mem_num,  device= x_spt.device), requires_grad=True)
        
        if os.path.isfile('fine.pt'):
            model.net.memory.We1 = torch.load('fine.pt')


        for ep in range(args.fine_epochs):
            loss_fine= []
            for data,label in val_loader_fine:
                sp_data, qu_data,sp_label,qu_label = split_batch(data,label) # make_datas(data,label)
                x_spt, y_spt, x_qry, y_qry = sp_data.to(args.device), sp_label.to(args.device), qu_data.to(args.device), qu_label.to(args.device)
                #print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape, graph_supports_fine.shape);exit()
                loss_f = model.finetunning(x_spt, y_spt, x_qry, y_qry, graph_supports_fine,
                cheb_polynomials_fine , L_tilde_fine)
                loss_fine.append(loss_f)
            
            print("train  epochs ",epoch, "fine epochs ", ep, "fine loss=",  np.mean(loss_fine))

        torch.save(model.net.memory.We1, 'fine.pt')
        model.net.memory.We1 = torch.load('train.pt')

'''






for epoch in range(args.epochs):
    
    loss_train= []
    for step, (data,label) in enumerate(train_loader):

        sp_data,sp_label, qu_data,qu_label = make_datas(data,label)
        x_spt, y_spt, x_qry, y_qry = sp_data.to(args.device), sp_label.to(args.device), qu_data.to(args.device), qu_label.to(args.device)
        #print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
        loss= model(x_spt, y_spt, x_qry, y_qry,   graph_supports, cheb_polynomials ,L_tilde)

        if epoch > 15:     
            torch.save(model.state_dict(), str(epoch)+"_best_model.pt")
            
        loss_train.append(loss)
        #break

        

        #print('step:', step, '\ttraining loss:', loss)
        # if step % 30 == 0:
        #     print('step:', step, '\ttraining loss:', loss)

        if step > 1 and step % 400 ==0 :
             # % 500 == 0:
            torch.save(model.net.memory.We1, 'train.pt')

            if not os.path.isfile('fine.pt'):
                node_num = 307
                model.net.memory.We1 = nn.Parameter(torch.randn(node_num, args.mem_num,  device= x_spt.device), requires_grad=True)
            
            if os.path.isfile('fine.pt'):
                model.net.memory.We1 = torch.load('fine.pt')


            loss_fine= []
            for data,label in val_loader_fine:
                sp_data, qu_data,sp_label,qu_label = split_batch(data,label) # make_datas(data,label)
                x_spt, y_spt, x_qry, y_qry = sp_data.to(args.device), sp_label.to(args.device), qu_data.to(args.device), qu_label.to(args.device)
                #print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape, graph_supports_fine.shape);exit()
                loss_f = model.finetunning(x_spt, y_spt, x_qry, y_qry, graph_supports_fine, cheb_polynomials_fine , L_tilde_fine)

                loss_fine.append(loss_f)
            
            print("fine loss", np.mean(loss_fine))
            torch.save(model.net.memory.We1, 'fine.pt')
            model.net.memory.We1 = torch.load('train.pt')

    print("train loss epoch", epoch, np.mean(loss_train))

