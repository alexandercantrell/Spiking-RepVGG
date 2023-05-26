import torch
import torch.nn as nn

def add_cnf(x,y):
    return torch.add(x,y)

def and_cnf(x,y):
    return nn.functional.relu(torch.subtract(torch.add(x,y),1))

def iand_cnf(x,y):
    return nn.functional.relu(torch.subtract(x,y))

def or_cnf(x,y):
    z = torch.add(x,y)
    return torch.where(z>1.0,1.0,z)

def xor_cnf(x,y):
    return torch.remainder(torch.add(x,y),2)

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        if cnf == 'ADD':
            self.cnf = lambda x,y: x+y
        elif cnf == 'AND':
            self.cnf = and_cnf
        elif cnf == 'IAND':
            self.cnf = iand_cnf
        elif cnf == 'OR':
            self.cnf = or_cnf
        elif cnf == 'XOR':
            self.cnf = xor_cnf
        else:
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
    def forward(self,x,y):
        return self.cnf(x,y)