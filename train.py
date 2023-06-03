
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import pickle

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class dfmodel(nn.Module):
    def __init__(self):
        super(dfmodel, self).__init__()
        self.representation = vip_tiny()
        self.linear = nn.Linear(2048,2)
    def forward(self,x):
        #shape of x should be [Batch, Frame, Channel, H, W]
        B = x.shape[0]
        out = torch.empty(B,8,32,512).to(device)
        for i in range(B):
            out_rep, _ = self.representation(x[i])
            out[i] = out_rep
        out = out.transpose(1,2)
        out = F.normalize(out, -1)
        out = torch.matmul(out, out.transpose(-1,-2)) #similarity matrix
        out = out.view(B,-1)
        out = self.linear(out)
        return out


model = dfmodel()
model.to(device)

optimizer = optim.Adam(model.parameters(), 
            lr = 0.001)

criterion = nn.CrossEntropyLoss()
min_valid_loss = np.inf
tl=[]
vl=[]
train_loss_file = open("train_loss.txt", "w")
val_file_loss = open("val_loss.txt", "w")
for epoch in range(100):
    
    train_loss= 0.0
    model.train()
    for inp, label in tqdm(train_loader):
        inp, label = inp.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, label.long().detach()) 

        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    tl.append(train_loss)
    train_loss_file.write(str(train_loss)+ "\n")
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)}')

    valid_loss = 0.0
    model.eval()
    for data, labels in val_loader:
        inp, label = inp.to(device), label.to(device)
        out = model(inp)
        loss = criterion(out, label.long().detach()) 
        valid_loss += loss.item() 

    val_file_loss.write(str(valid_loss)+ "\n")
    vl.append(valid_loss)
    print(f'Epoch {epoch+1} \t\t Validation Loss: {valid_loss / len(val_loader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'best_model.pth')



