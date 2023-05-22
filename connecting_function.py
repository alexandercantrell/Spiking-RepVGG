import torch
import torch.nn as nn

def add_cnf(x,y):
    return x+y

def and_cnf(x,y):
    return x*y

def where_and_cnf(x,y):
    z = x+y
    return torch.where(z>1.0,1.0,0.0)

def iand_cnf(x,y):
    return x * (1. - y) 

def where_iand_cnf(x,y):
    z = x + 1. - y
    return torch.where(z>1.0,1.0,0.0)

def or_cnf(x,y):
    return x+y-(x*y)

def where_or_cnf(x,y):
    z = x+y
    return torch.where(z>0.0,1.0,z)

def xor_cnf(x,y):
    return (x+y)%2

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        if cnf == 'ADD':
            self.cnf = lambda x,y: x+y
        elif cnf == 'AND':
            self.cnf = and_cnf
        elif cnf == 'WAND':
            self.cnf = where_and_cnf
        elif cnf == 'IAND':
            self.cnf = iand_cnf
        elif cnf == 'WIAND':
            self.cnf = where_iand_cnf
        elif cnf == 'OR':
            self.cnf = or_cnf
        elif cnf == 'WOR':
            self.cnf = where_or_cnf
        elif cnf == 'XOR':
            self.cnf = xor_cnf
        else:
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
    def forward(self,x,y):
        return self.cnf(x,y)