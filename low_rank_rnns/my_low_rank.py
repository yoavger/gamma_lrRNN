import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Module

from low_rank_rnns.helpers import *
import torch.nn as nn
from math import sqrt, floor
import random
import time


'''
============================================

Description : PyTorch autograd function gamma(x;n,s) parametrised by :
    n > 0      : neuronal gain
    s in [0,1] : degree of saturation

============================================
'''

# ---------------------------
#   Heterogeneous adaptation
# ---------------------------

def batch_softplus(x, beta, threshold):
    '''
    Softplus reformulation for overflow handling.
    '''
    lins = torch.mul(x, beta)
    val = torch.div(torch.log(1 + torch.exp(torch.mul(beta,x))), beta)
    val = torch.where(lins > threshold, torch.mul(beta.sign(), x), val)
    return val.t()

class Gamma2(torch.autograd.Function):
    '''
    Gamma autograd function for heteogeneous adaptation

    Forward params : 
    - input : torch tensor of shape (batch_size, input_dimension)
    - n     : neuronal gain, torch tensor of shape (input_dimension,)
    - s     : saturaiton, torch tensor of shape (input_dimension,) 
    '''
    @staticmethod
    def forward(ctx, input, n, s):
        ctx.n = n
        ctx.s = s

        gamma_one = batch_softplus(input, n, 20)
        gamma_two = torch.sigmoid(torch.mul(n, input).t())
        output = torch.mul((1-s), gamma_one.t()) + torch.mul(s, gamma_two.t())

        ctx.save_for_backward(input, gamma_one.t(), gamma_two.t())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma_one, gamma_two = ctx.saved_tensors
        
        n = ctx.n
        s = ctx.s

        grad_input = grad_n = grad_s = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * ((1-s)*gamma_two + s*n*gamma_two*(1-gamma_two))
        if ctx.needs_input_grad[1]:
            grad_n = grad_output * ((1-s)/n * (input * gamma_two - gamma_one) + s*input*gamma_two*(1-gamma_two))
        if ctx.needs_input_grad[2]:
            grad_s = grad_output * (gamma_two - gamma_one)

        return grad_input, grad_n, grad_s

class gamma2(nn.Module):
    def __init__(self, n, s, hidden_size, random_init=False):
        super(gamma2, self).__init__()
        self.n = n
        self.s = s
        if type(n)==int or type(n)==float:
            self.n = n*torch.ones(hidden_size)

        if type(s)==int or type(s)==float:
            self.s = s*torch.ones(hidden_size)

    def forward(self, input):
        #print(self.n.shape)
        return Gamma2.apply(input, self.n, self.s)

class batch_gamma(nn.Module):
    def __init__(self, n, s, hidden_size, batchsize=100, random_init=False):
        super(batch_gamma, self).__init__()
        self.n = n
        self.s = s

        # print(self.n.shape)
        self.n = n.repeat(hidden_size, 1).t()
        # print(self.n.shape)
        self.s = s.repeat(hidden_size, 1).t()

        if type(n)==int or type(n)==float:
            self.n = n*torch.ones(hidden_size)

        if type(s)==int or type(s)==float:
            self.s = s*torch.ones(hidden_size)

    def forward(self, input):
        return Gamma2.apply(input, self.n, self.s)

class batch_gamma2(nn.Module):
    def __init__(self, n, s, hidden_size, batchsize=100, random_init=False):
        super(batch_gamma, self).__init__()
        self.n = n
        self.s = s

        # print(self.n.shape)
        self.n = n.repeat(hidden_size, 1).t()
        # print(self.n.shape)
        self.s = s.repeat(hidden_size, 1).t()

        if type(n)==int or type(n)==float:
            self.n = n*torch.ones(hidden_size)

        if type(s)==int or type(s)==float:
            self.s = s*torch.ones(hidden_size)

    def forward(self, input):
        return Gamma2.apply(input, self.n, self.s)
    
    
class myLowRankRNN(nn.Module):
    """
    This class implements the low-rank RNN. Instead of being parametrized by an NxN connectivity matrix, it is
    parametrized by two Nxr matrices m and n such that the connectivity is m * n^T
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, si_init=None, so_init=None, h0_init=None,
                 add_biases=False, non_linearity=torch.tanh,output_non_linearity=torch.tanh, gain_init=None, sat_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rank: int
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool
        :param train_so: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param m_init: torch tensor of shape (hidden_size, rank)
        :param n_init: torch tensor of shape (hidden_size, rank)
        :param si_init: torch tensor of shape (input_size)
        :param so_init: torch tensor of shape (output_size)
        :param h0_init: torch tensor of shape (hidden_size)
        """
        super(myLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = non_linearity
        self.output_non_linearity = output_non_linearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.m = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, rank))
        if not train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        if not add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False
            
        learn_params = True 
        ###############################################
        if learn_params:
            self.gain = nn.Parameter(torch.rand(self.hidden_size)*10)
            self.sat = nn.Parameter(torch.rand(self.hidden_size))
        else:
            self.gain = torch.tensor([gain_init])
            self.sat = torch.tensor([sat_init])
        ###############################################

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if m_init is None:
                self.m.normal_()
            else:
                self.m.copy_(m_init)
            if n_init is None:
                self.n.normal_()
            else:
                self.n.copy_(n_init)
            self.b.zero_()     # TODO add biases initializer
            if wo_init is None:
                self.wo.normal_(std=4.)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
            
            if gain_init is None:
                self.gain = nn.Parameter(torch.rand(self.hidden_size)*10)
            else:
                self.gain.copy_(gain_init)
            
            if sat_init is None:
                self.sat = nn.Parameter(torch.rand(self.hidden_size))
            else:
                self.sat.copy_(sat_init)
                
        self.non_linearity = gamma2(self.gain, self.sat, self.hidden_size)
                
        self.wrec, self.wi_full, self.wo_full = [None] * 3
        self._define_proxy_parameters()
        
    def _define_proxy_parameters(self):
        self.wrec = None   # For optimization purposes the full connectivity matrix is never computed explicitly
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so
                
    def forward(self, input, return_dynamics=False, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: boolean
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
            
        h = initial_states.clone()
        
        with_gain = True
        
        ###############################################
        if with_gain:
#             nonlinearity = gamma2(self.gain, self.sat, self.hidden_size)
#             r = nonlinearity(h)
            r = self.non_linearity(h)
        else:
            r = self.non_linearity(h)
        ###############################################

        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.n).matmul(self.m.t()) / self.hidden_size +
                    input[:, i, :].matmul(self.wi_full))
            
            ###############################################
            if with_gain:
#                 nonlinearity = gamma2(self.gain, self.sat, self.hidden_size)
#                 r = nonlinearity(h+self.b)
                r = self.non_linearity(h + self.b)
            else:
                r = self.non_linearity(h + self.b)
            ###############################################
            
            output[:, i, :] = self.output_non_linearity(h) @ self.wo_full / self.hidden_size
            if return_dynamics:
                trajectories[:, i + 1, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = myLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             self.rank, self.train_wi, self.train_wo, self.train_wrec, self.train_h0, self.train_si,
                             self.train_so, self.wi, self.wo, self.m, self.n, self.si, self.so, self.h0, False,
                             self.non_linearity, self.output_non_linearity, self.gain, self.sat)
        new_net._define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override
        """
        if 'rec_noise' in state_dict:
            del state_dict['rec_noise']
        super().load_state_dict(state_dict, strict)
        self._define_proxy_parameters()

    def svd_reparametrization(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self._define_proxy_parameters()

            