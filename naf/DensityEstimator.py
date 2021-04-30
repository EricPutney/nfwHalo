#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:45:34 2018

@author: chin-weihuang
"""


from torchkit import flows, nn as nn_, utils
from torch import optim, nn
from torch.autograd import Variable
import torch
import os
import numpy as np
import math


class DensityEstimator(object):
    
    def __init__(self, flowtype=0, dim=2, dimh=100, 
                 n=128, num_hid_layers=2,
                 act=nn.ELU(), num_flow_layers=5, 
                 num_ds_dim=16, num_ds_layers=2,
                 lr=0.000002, betas=(0.9,0.999), clip=5.0,fixed_order=True,
                 calc_grad=False):
        
        self.calc_grad = calc_grad
        
        if flowtype == 0:
            flow = flows.IAF.cuda()
            
        elif flowtype == 1:
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs).cuda()

        sequels = [nn_.SequentialFlow(
            flow(dim=dim,
                 hid_dim=dimh,
                 context_dim=1,
                 num_layers=num_hid_layers+1,
                 activation=act,
                 fixed_order=fixed_order),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(dim, 1),]
        
        self.mdl = nn.Sequential(
                *sequels).cuda()
                
        self.optim = optim.Adam(self.mdl.parameters(), lr=lr, betas=betas)
            
        self.n = n
        self.dim=dim
        self.clip=clip
        
    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.mdl.parameters(),
                                self.clip)
        
    def density(self, spl):
        n = spl.size(0)
        
        context = Variable(torch.FloatTensor(n, 1).zero_()) 
        lgd = Variable(torch.FloatTensor(n).zero_())
        zeros = Variable(torch.FloatTensor(n, self.dim).zero_())
        context = context.cuda()
        lgd = lgd.cuda()
        zeros = zeros.cuda()
        
        z, logdet, _ = self.mdl((spl, lgd, context))
        losses = - utils.log_normal(z, zeros, zeros+1.0).sum(1) - logdet
        return - losses
    
    def fit(self, dataloader, save_directory=".",total=2000, verbose=True):
        
#        sampler = distr.sampler
#        n = self.n
        
        for it in range(total):

            trainloss=0.
            ndata=0    
            for batch_idx, data in enumerate(dataloader):
                self.optim.zero_grad()
#                print('data: ',data)
                losses = - self.density(data.cuda().float())
#                print('losses: ',losses)
                loss = losses.mean()

                trainloss+=losses.sum()
                ndata+=len(data)

                loss.backward()
                self.clip_grad_norm()

                self.optim.step()

                
            if verbose:
                print('Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, (trainloss.data.item())/ndata))
            
                torch.save(self.mdl.state_dict(), save_directory+"nafmodel_"+str(it)+".dict")

    def return_grad(self, data):
        self.optim.zero_grad()
            
        data = Variable(data,requires_grad=True)
        losses = - self.density(data.cuda().float())
        gradients, = torch.autograd.grad(losses, data,  grad_outputs=torch.ones_like(losses),
                                         retain_graph=True,create_graph=True)

        acceleration, = torch.autograd.grad(gradients, data, grad_outputs=torch.ones_like(gradients),
                                            retain_graph=True,create_graph=True)

        dim = data.shape[-1]
        length = data.shape[0]
        acc_tensor = torch.ones((length,dim,dim))
        for i in range(dim):
            acc, = torch.autograd.grad(gradients[:,i], data, grad_outputs=torch.ones_like(gradients[:,i]),
                                            retain_graph=True,create_graph=True)
            acc = -acc+gradients[:,i,None]*gradients
            acc_tensor[:,i,:] = acc
            
        
#        self.clip_grad_norm()
            
#        self.optim.step()

#        return -gradients, -acceleration+gradients*gradients
        return -gradients, acc_tensor
