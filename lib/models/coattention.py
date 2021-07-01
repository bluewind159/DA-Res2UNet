import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn.functional as F
import torch.nn as nn
class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)   

class coattention(nn.Module):
    def __init__(self,hidden_dim0,hidden_dim1):
        super(coattention,self).__init__()
        d=hidden_dim0
        g=hidden_dim1
        k=512
        m=512
        n=512
        self.get_c=Linear(g,d,bias=False)
        self.get_Hv=Linear(g,k,bias=False)
        self.get_Hq=Linear(d,k,bias=False)
        self.get_att_Q=Linear(k,d,bias=False)
        self.get_att_V=Linear(k,g,bias=False)
    def forward(self,Q,V):
        asd=self.get_c(V)
        C=F.tanh(torch.matmul(Q,asd.transpose(2,1)))
        Hv=F.tanh(self.get_Hv(V).transpose(2,1)+torch.matmul(self.get_Hq(Q).transpose(2,1),C))
        Hq=F.tanh(self.get_Hq(Q).transpose(2,1)+torch.matmul(self.get_Hv(V).transpose(2,1),C.transpose(2,1)))
        att_V=F.softmax(self.get_att_V(Hv.transpose(2,1)),dim=-1)
        att_Q=F.softmax(self.get_att_Q(Hq.transpose(2,1)),dim=-1)
        Q=Q*att_Q
        V=V*att_V
        return Q,V
        
