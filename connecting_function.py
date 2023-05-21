import torch.nn as nn

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        if cnf == 'ADD':
            self.cnf = lambda x,y: x+y
        elif cnf == 'AND':
            self.cnf = lambda x,y:  x * y
        elif cnf == 'IAND':
            self.cnf = lambda x,y: x * (1. - y) 
        elif cnf == 'OR':
            self.cnf = lambda x,y: x+y-(x*y)
        elif cnf == 'XOR':
            self.cnf = lambda x,y: x+y-(2*x*y)
        else:
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
    def forward(self,x,y):
        return self.cnf(x,y)