import torch,pdb,math
import torch.nn as nn
import torch.nn.functional as F
from argu import args
import pdb

class AttLayer(nn.Module):
    def __init__(self, out_channels, use_bias=False, reduction=16):
        super(AttLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, 1, bias=False),
            nn.Sigmoid()
            #nn.Hardsigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1)
        # y = F.hardsigmoid(self.fc(y)).view(b, 1, 1)
        return x * y.expand_as(x)


class cheb_conv(nn.Module):
    
    def __init__(self,cheb_k, dim_in, dim_out):
        super(cheb_conv,self).__init__()
        self.gconv = nn.Linear(dim_out * cheb_k, dim_out)
        self.init_gconv = nn.Linear(dim_in, dim_out)


    def forward(self,x,graph_supports):
        b, n, _ = x.shape
        x = self.init_gconv(x)
        #pdb.set_trace()
        x_g1 = torch.einsum("knm,bmc->bknc", graph_supports, x)
        x_g1 = x_g1.permute(0, 2, 1, 3).reshape(b, n, -1)  # B, N, cheb_k, dim_in
        x_gconv1 =self.gconv(x_g1)

        return x_gconv1


class gcn_glu(nn.Module):
    def __init__(self,c_in,c_out):
        super(gcn_glu,self).__init__()
        self.mlp =torch.nn.Conv2d(c_in, 2*c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.c_out = c_out

    def forward(self, x , graph_supports):
        
        # (N, B, C)
        x = x.unsqueeze(3) # (N, B, C, 1)
        x = x.permute(1, 2, 0, 3) # (N, B, C, 1)->(B, C, N, 1)
        ax = torch.einsum('vn,bfnt->bfvt',(graph_supports,x)).contiguous()
        axw = self.mlp(ax) # (B, 2C', N, 1)
        axw_1,axw_2 = torch.split(axw, [self.c_out, self.c_out], dim=1)
        axw_new = axw_1 * torch.sigmoid(axw_2) # (B, C', N, 1)
        axw_new = axw_new.squeeze(3) # (B, C', N)
        axw_new = axw_new.permute(0,2,1) # ( B,N, C')
        return axw_new
    


class RGSLCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(RGSLCell, self).__init__()
        #pdb.set_trace()
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN( dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, learned_tilde,cheb_polynomials, L_tilde):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim

        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, learned_tilde,cheb_polynomials, L_tilde))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, learned_tilde, cheb_polynomials, L_tilde))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)
    
class AVWGCN(nn.Module):
    def __init__(self,  dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        #self.cheb_polynomials = cheb_polynomials
        #self.L_tilde = L_tilde
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        
        # for existing graph convolution
        # self.init_gconv = nn.Conv1d(dim_in, dim_out, kernel_size=5, padding=0)
        self.init_gconv = nn.Linear(dim_in, dim_out)
        self.gconv = nn.Linear(dim_out * cheb_k, dim_out)
        self.dy_gate1 = AttLayer(dim_out)
        self.dy_gate2 = AttLayer(dim_out)

        self.dim_in= dim_in
        self.dim_out= dim_out

    def forward(self, x, node_embeddings, L_tilde_learned,cheb_polynomials, L_tilde):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        b, n, _ = x.shape
        # 0) learned cheb_polynomials
        node_num = node_embeddings.shape[0]

        # L_tilde_learned = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # L_tilde_learned = torch.matmul(L_tilde_learned, self.L_tilde) * L_tilde_learned

        support_set = [torch.eye(node_num).to(L_tilde_learned.device), L_tilde_learned]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * L_tilde_learned, support_set[-1]) - support_set[-2])

        # 1) convolution with learned graph convolution (implicit knowledge)
        #pdb.set_trace()
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out

        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv0 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out

        # 2) convolution with existing graph (explicit knowledge)
        #import pdb; pdb.set_trace()
        graph_supports = torch.stack(cheb_polynomials, dim=0)  # [k, n, m]
        x = self.init_gconv(x)
        x_g1 = torch.einsum("knm,bmc->bknc", graph_supports, x)
        x_g1 = x_g1.permute(0, 2, 1, 3).reshape(b, n, -1)  # B, N, cheb_k, dim_in
        x_gconv1 = self.gconv(x_g1)

        # 3) fusion of explit knowledge and implicit knowledge
        x_gconv = self.dy_gate1(F.leaky_relu(x_gconv0).transpose(1,2)) + self.dy_gate2(F.leaky_relu(x_gconv1).transpose(1,2))
        # x_gconv = F.leaky_relu(x_gconv0) + F.leaky_relu(x_gconv1)
        
        return x_gconv.transpose(1,2)



class AVWDCRNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        # print(dim_in, dim_out, "sdfdfsdfsdfdf")
        self.dcrnn_cells.append(RGSLCell(  dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(RGSLCell( dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, learned_tilde,cheb_polynomials, L_tilde):
        #pdb.set_trace()
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[3] == self.input_dim # x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        #print(x.shape)
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, learned_tilde,cheb_polynomials, L_tilde)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            #print(current_inputs.shape)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size,num_nodes):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size,num_nodes))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)



class WeeklyPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(WeeklyPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=0.1)
        self.register_buffer('positional_encoding', self._create_encoding())

    def _create_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        weekly_freq = 2 * math.pi / (7 * 24)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * weekly_freq)
        pe[:, 1::2] = torch.cos(position * weekly_freq)
        return pe.unsqueeze(0)

    def forward(self, x):
        x =x.squeeze(3).permute(0,2,1)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x =self.dropout(x)
        x= x.permute(0,2,1).unsqueeze(3)
        return x


class DailyPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(DailyPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=0.1)
        self.register_buffer('positional_encoding', self._create_encoding())

    def _create_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        daily_freq = 2 * math.pi / 24
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * daily_freq)
        pe[:, 1::2] = torch.cos(position * daily_freq)
        return pe.unsqueeze(0)

    def forward(self, x):
        x=x.squeeze(3).permute(0,2,1)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x =self.dropout(x)
        x=x.permute(0,2,1).unsqueeze(3)
        return x
    


# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super(LearnablePositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_sequence_length = max_seq_len
#         self.embedding = nn.Embedding(max_seq_len, d_model)

#     def forward(self, x):
#         x=x.squeeze(3).permute(0,2,1)
#         positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
#         pos_enc = self.embedding(positions)
#         x += pos_enc
#         return x.permute(0,2,1).unsqueeze(3) 
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_embedding = nn.Embedding(max_length+93, embedding_dim)
        self.scale = nn.Parameter(torch.ones(1, 1, embedding_dim))
    
    def forward(self, x,adj_inp,cheb_polynomials, L_tilde, eval=False):
        x=x.squeeze(3).permute(0,2,1)
        batch_size, seq_len, input_dim = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embedding = self.positional_embedding(positions)
        #pdb.set_trace()
        encoded_x = x + (self.scale * pos_embedding) #.unsqueeze(2)
        encoded_x=encoded_x.permute(0,2,1).unsqueeze(3)
        return encoded_x
    

    # def functional_forward(self,x,weight):

    #     x=x.squeeze(3).permute(0,2,1)
    #     batch_size, seq_len, input_dim = x.size()
    #     positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    #     pos_embedding= F.embedding(positions, weight['positional_embedding.weight'])
    #     encoded_x = x + (weight['scale'] * pos_embedding)

    #     return encoded_x.permute(0,2,1).unsqueeze(3)


