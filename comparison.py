import torch
import numpy as np
import math
import tensorly as tl
import tntorch as tn

def function(X):  # Matrix (one row per sample, one column per input variable) and return a vector with one result per sample
    return torch.sin(torch.sum(X,dim=1))  #f(x1,...,xN) = sin(x1+...+xN)

domain = [torch.arange(1, 33) for n in range(3)]

t = tn.cross(function = function, domain=domain,function_arg='matrix',ranks_tt=3,rmax=100,return_info=True)
print(t)

def f(X):
    return 1/torch.sum(X,dim=1) #Hilbert tensor

domain = [torch.arange(1, 33) for n in range(2)]

s = tn.cross(function = f, domain=domain,function_arg='matrix',ranks_tt=5,rmax=100)
print(s)