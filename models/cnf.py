import torch
from torch import nn 
from torch.autograd import Function

def add_cnf(x,y):
    return x+y

def and_cnf(x,y):
    return x*y

def iand_cnf(x,y):
    return x * (1.-y)

class IAND(Function):
    @staticmethod
    def forward(x,y):
        return nn.functional.relu(x-y)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-y)*grad_output
        grad_y = -x * grad_output
        return grad_x, grad_y

def fast_iand_cnf(x,y):
    return IAND.apply(x,y)

def nand_cnf(x,y):
    return 1.-x*y

class NAND(Function):
    @staticmethod
    def forward(x,y):
        return 1.-x*y
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = -y * grad_output
        grad_y = -x * grad_output
        return grad_x, grad_y
    
def fast_nand_cnf(x,y):
    return NAND.apply(x,y)

def or_cnf(x,y):
    return x+y-x*y

class OR(Function):
    @staticmethod
    def forward(x,y):
        return x+y-x*y
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-y) * grad_output
        grad_y = (1.-x) * grad_output
        return grad_x, grad_y

def fast_or_cnf(x,y):
    return OR.apply(x,y)    

def nor_cnf(x,y):
    return 1.-(x+y-x*y)

class NOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x, y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output
        grad_y = (x-1.) * grad_output
        return grad_x, grad_y
    
def fast_nor_cnf(x,y):
    return NOR.apply(x,y)

def xor_cnf(x,y):
    return x+y-(2.*x*y)

class XOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x, y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y))*grad_output
        grad_y = (1.-(2.*x))*grad_output
        return grad_x, grad_y
    
def fast_xor_cnf(x,y):
    return XOR.apply(x,y)
    
class WeightedXOR(XOR):
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        base = torch.sub(1.,torch.mul(x,y))
        grad_x = (base-(2.*y))*grad_output
        grad_y = (base-(2.*x))*grad_output
        return grad_x, grad_y
    
def weighted_xor_cnf(x,y):
    return WeightedXOR.apply(x,y)

def xnor_cnf(x,y):
    return 1.-(x+y-(2.*x*y))

class XNOR(Function):
    @staticmethod
    def forward(x,y):
        return 1.-((x+y)%2)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x, y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = ((2.*y)-1.)*grad_output
        grad_y = ((2.*x)-1.)*grad_output
        return grad_x, grad_y
    
def fast_xnor_cnf(x,y):
    return XNOR.apply(x,y)

CONNECTING_FUNCTIONS={
    'ADD':add_cnf,
    'AND':and_cnf,
    'IAND':iand_cnf,
    'FAST_IAND':fast_iand_cnf,
    'NAND':nand_cnf,
    'FAST_NAND':fast_nand_cnf,
    'OR':or_cnf,
    'FAST_OR':fast_or_cnf,
    'NOR':nor_cnf,
    'FAST_NOR':fast_nor_cnf,
    'XOR': xor_cnf,
    'FAST_XOR':fast_xor_cnf,
    'WEIGHTED_XOR':weighted_xor_cnf,
    'XNOR':xnor_cnf,
    'FAST_XNOR': fast_xnor_cnf
}

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        if cnf not in CONNECTING_FUNCTIONS.keys():
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
        self.cnf=CONNECTING_FUNCTIONS[cnf]
    def forward(self,x,y):
        return self.cnf(x,y)