import torch
from torch import nn 
from torch.autograd import Function

def add_cnf(x,y):
    return x+y

def and_cnf(x,y):
    return x*y

class SURROGATE_AND(Function):
    @staticmethod
    def forward(x,y):
        return x*y
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = y.where(x!=y,1.) * grad_output
        grad_y = x.where(x!=y,1.) * grad_output
        return grad_x, grad_y
    
def surrogate_and_cnf(x,y):
    return SURROGATE_AND.apply(x,y)

class CONST_SURROGATE_AND(Function):
    @staticmethod
    def forward(x,y):
        return x*y
    @staticmethod
    def setup_context(ctx,inputs,output):
        pass
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output, grad_output
    
def const_surrogate_and_cnf(x,y):
    return CONST_SURROGATE_AND.apply(x,y)

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

class SURROGATE_IAND(Function):
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
        grad_x = (1.-y).where(x==y,1.) * grad_output
        grad_y = -x.where(x==y,1.) * grad_output
        return grad_x, grad_y
    
def surrogate_iand_cnf(x,y):
    return SURROGATE_IAND.apply(x,y)

class CONST_SURROGATE_IAND(Function):
    @staticmethod
    def forward(x,y):
        return nn.functional.relu(x-y)
    @staticmethod
    def setup_context(ctx,inputs,output):
        pass
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output, -grad_output
    
def const_surrogate_iand_cnf(x,y):
    return CONST_SURROGATE_IAND.apply(x,y)
    
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

class SURROGATE_NAND(Function):
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
        grad_x = -y.where(x==y,1.) * grad_output
        grad_y = -x.where(x==y,1.) * grad_output
        return grad_x, grad_y

def surrogate_nand_cnf(x,y):
    return SURROGATE_NAND.apply(x,y)

class CONST_SURROGATE_NAND(Function):
    @staticmethod
    def forward(x,y):
        return 1.-x*y
    @staticmethod
    def setup_context(ctx,inputs,output):
        pass
    @staticmethod
    def backward(ctx,grad_output):
        return -grad_output, -grad_output
    
def const_surrogate_nand_cnf(x,y):
    return CONST_SURROGATE_NAND.apply(x,y)

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

class SURROGATE_OR(Function):
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
        grad_x = (1.-y).where(x!=y,1.) * grad_output
        grad_y = (1.-x).where(x!=y,1.) * grad_output
        return grad_x, grad_y
    
def surrogate_or_cnf(x,y):
    return SURROGATE_OR.apply(x,y)

class CONST_SURROGATE_OR(Function):
    @staticmethod
    def forward(x,y):
        return x+y-x*y
    @staticmethod
    def setup_context(ctx,inputs,output):
        pass
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output, grad_output
    
def const_surrogate_or_cnf(x,y):
    return CONST_SURROGATE_OR.apply(x,y)

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

class SURROGATE_NOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1.) * grad_output
        grad_y = (x-1.).where(x!=y,-1.) * grad_output
        return grad_x, grad_y
    
def surrogate_nor_cnf(x,y):
    return SURROGATE_NOR.apply(x,y)

class CONST_SURROGATE_NOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        pass
    @staticmethod
    def backward(ctx,grad_output):
        return -grad_output, -grad_output
    
def const_surrogate_nor_cnf(x,y):
    return CONST_SURROGATE_NOR.apply(x,y)

class RIGHT_NOR(Function):
    #only surrogate for y value
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_y = None
        grad_y = (x-1.) * grad_output
        return torch.zeros(grad_y.shape,device=grad_y.device), grad_y

def right_nor_cnf(x,y):
    return RIGHT_NOR.apply(x,y)
    
class RIGHT_SURROGATE_NOR(Function):
    #only surrogate for y value
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_y = None
        grad_y = (x-1.).where(x!=y,-1.) * grad_output
        return torch.zeros(grad_y.shape,device=grad_y.device), grad_y
    
def right_surrogate_nor_cnf(x,y):
    return RIGHT_SURROGATE_NOR.apply(x,y)

class SUB_NOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return 1.-(x+y-x*y)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output
        grad_y = (x-2.) * grad_output
        return grad_x, grad_y
    
def sub_nor_cnf(x,y):
    return SUB_NOR.apply(x,y)

class SUB_SURROGATE_NOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1.) * grad_output
        grad_y = (x-2.).where(x!=y,-2.) * grad_output 
        return grad_x, grad_y
    
def sub_surrogate_nor_cnf(x,y):
    return SUB_SURROGATE_NOR.apply(x,y)

class ADD_NOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output
        grad_y = x * grad_output 
        return grad_x, grad_y
    
def add_nor_cnf(x,y):
    return ADD_NOR.apply(x,y)

class ADD_SURROGATE_NOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1) * grad_output
        grad_y = x.where(x!=y,0) * grad_output 
        return grad_x, grad_y
    
def add_surrogate_nor_cnf(x,y):
    return ADD_SURROGATE_NOR.apply(x,y)

class DOUBLE_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output
        grad_y = (x-1.) * grad_output * 2.
        return grad_x, grad_y

def double_nor_cnf(x,y):
    return DOUBLE_NOR.apply(x,y)

class DOUBLE_SURROGATE_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1.) * grad_output
        grad_y = (x-1.).where(x!=y,-1.) * grad_output * 2.
        return grad_x, grad_y
    
def double_surrogate_nor_cnf(x,y):
    return DOUBLE_SURROGATE_NOR.apply(x,y)

class LEFT_CONSTANT_NOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = - grad_output
        grad_y = (x-1.) * grad_output
        return grad_x, grad_y
    
def left_constant_nor_cnf(x,y):
    return LEFT_CONSTANT_NOR.apply(x,y)

class LEFT_CONSTANT_SURROGATE_NOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = - grad_output
        grad_y = (x-1.).where(x!=y,-1.) * grad_output
        return grad_x, grad_y
    
def left_constant_surrogate_nor_cnf(x,y):
    return LEFT_CONSTANT_SURROGATE_NOR.apply(x,y)

class COMBO_NOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = y * grad_output
        grad_y = x * grad_output
        return grad_x, grad_y

def combo_nor_cnf(x,y):
    return COMBO_NOR.apply(x,y)

class COMBO_SURROGATE_NOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = y.where(x!=y,0) * grad_output
        grad_y = x.where(x!=y,0) * grad_output
        return grad_x, grad_y
    
def combo_surrogate_nor_cnf(x,y):
    return COMBO_SURROGATE_NOR.apply(x,y)

class HALF_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output / 2.
        grad_y = (x-1.) * grad_output 
        return grad_x, grad_y

def half_nor_cnf(x,y):
    return HALF_NOR.apply(x,y)

class HALF_SURROGATE_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1.) * grad_output / 2.
        grad_y = (x-1.).where(x!=y,-1.) * grad_output 
        return grad_x, grad_y
    
def half_surrogate_nor_cnf(x,y):
    return HALF_SURROGATE_NOR.apply(x,y)

class LEFT_DOUBLE_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output
        grad_y = (x-1.) * grad_output / 2.
        return grad_x, grad_y

def left_double_nor_cnf(x,y):
    return LEFT_DOUBLE_NOR.apply(x,y)

class LEFT_DOUBLE_SURROGATE_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1.) * grad_output
        grad_y = (x-1.).where(x!=y,-1.) * grad_output / 2.
        return grad_x, grad_y
    
def left_double_surrogate_nor_cnf(x,y):
    return LEFT_DOUBLE_SURROGATE_NOR.apply(x,y)

class BOTH_HALF_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.) * grad_output / 2.
        grad_y = (x-1.) * grad_output / 2.
        return grad_x, grad_y
    
def both_half_nor_cnf(x,y):
    return BOTH_HALF_NOR.apply(x,y)

class BOTH_HALF_SURROGATE_NOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y==0.).to(x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (y-1.).where(x!=y,-1.) * grad_output
        grad_y = (x-1.).where(x!=y,-1.) * grad_output / 2.
        return grad_x, grad_y

def both_half_surrogate_nor_cnf(x,y):
    return BOTH_HALF_SURROGATE_NOR.apply(x,y)
    
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
    
class SURROGATE_XOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y)).where(x==y,x) * grad_output
        grad_y = (1.-(2.*x)).where(x==y,y) * grad_output
        return grad_x, grad_y
    
def surrogate_xor_cnf(x,y):
    return SURROGATE_XOR.apply(x,y)

class RIGHT_XOR(Function):
    #only surrogate for y value
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_y = None
        grad_y = (1.-(2.*x)) * grad_output
        return torch.zeros(grad_y.shape,device=grad_y.device), grad_y
    
def right_xor_cnf(x,y):
    return RIGHT_XOR.apply(x,y)

class RIGHT_SURROGATE_XOR(Function):
    #only surrogate for y value
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_y = None
        grad_y = (1.-(2.*x)).where(x==y,y) * grad_output
        return torch.zeros(grad_y.shape,device=grad_y.device), grad_y
    
def right_surrogate_xor_cnf(x,y):
    return RIGHT_SURROGATE_XOR.apply(x,y)

class ADD_XOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y)) * grad_output
        grad_y = (2.-(2.*x)) * grad_output 
        return grad_x, grad_y
    
def add_xor_cnf(x,y):
    return ADD_XOR.apply(x,y)

class ADD_SURROGATE_XOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y)).where(x==y,x) * grad_output 
        grad_y = (2.-(2.*x)).where(x==y,y+1) * grad_output 
        return grad_x, grad_y

def add_surrogate_xor_cnf(x,y):
    return ADD_SURROGATE_XOR.apply(x,y)

class DOUBLE_XOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y)) * grad_output
        grad_y = (1.-(2.*x)) * grad_output * 2.
        return grad_x, grad_y

def double_xor_cnf(x,y):
    return DOUBLE_XOR.apply(x,y)

class DOUBLE_SURROGATE_XOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y)).where(x==y,x) * grad_output
        grad_y = (1.-(2.*x)).where(x==y,y) * grad_output * 2.
        return grad_x, grad_y
    
def double_surrogate_xor_cnf(x,y):
    return DOUBLE_SURROGATE_XOR.apply(x,y)

class LEFT_CONSTANT_XOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = grad_output
        grad_y = (1.-(2.*x)) * grad_output
        return grad_x, grad_y
    
def left_constant_xor_cnf(x,y):
    return LEFT_CONSTANT_XOR.apply(x,y)

class LEFT_CONSTANT_SURROGATE_XOR(Function):
    #only surrogate for x value
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = grad_output
        grad_y = (1.-(2.*x)).where(x==y,y) * grad_output
        return grad_x, grad_y
    
def left_constant_surrogate_xor_cnf(x,y):
    return LEFT_CONSTANT_SURROGATE_XOR.apply(x,y)

class COMBO_XOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (2.-(2.*y)) * grad_output
        grad_y = (2.-(2.*x)) * grad_output
        return grad_x, grad_y
    
def combo_xor_cnf(x,y):
    return COMBO_XOR.apply(x,y)

class COMBO_SURROGATE_XOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (2.-(2.*y)).where(x==y,x+1) * grad_output
        grad_y = (2.-(2.*x)).where(x==y,y+1) * grad_output
        return grad_x, grad_y
    
def combo_surrogate_xor_cnf(x,y):
    return COMBO_SURROGATE_XOR.apply(x,y)

class SPECIAL_XOR(Function):
    #surrogate for both x and y values
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = x.where(x!=y,0) * grad_output
        grad_y = (1.-(2.*x)).where(grad_x==0,0) * grad_output
        return grad_x, grad_y
    
def special_xor_cnf(x,y):
    return SPECIAL_XOR.apply(x,y)

class BOTH_HALF_XOR(Function):
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
        grad_x = (1.-(2.*y))*grad_output/2
        grad_y = (1.-(2.*x))*grad_output/2
        return grad_x, grad_y
    
def both_half_xor_cnf(x,y):
    return BOTH_HALF_XOR.apply(x,y)
    
class BOTH_HALF_SURROGATE_XOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = (1.-(2.*y)).where(x==y,x) * grad_output / 2
        grad_y = (1.-(2.*x)).where(x==y,y) * grad_output / 2
        return grad_x, grad_y
    
def both_half_surrogate_xor_cnf(x,y):
    return BOTH_HALF_SURROGATE_XOR.apply(x,y)

class XDEP_XOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = x * grad_output
        grad_y = (1-x)*grad_output
        return grad_x, grad_y
    
def xdep_xor_cnf(x,y):
    return XDEP_XOR.apply(x,y)

class YDEP_XOR(Function):
    @staticmethod
    def forward(x,y):
        return (x+y)%2
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
        return None
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_y = y * grad_output
        grad_x = (1-y)*grad_output
        return grad_x, grad_y
    
def ydep_xor_cnf(x,y):
    return YDEP_XOR.apply(x,y)

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

class SURROGATE_XNOR(Function):
    @staticmethod
    def forward(x,y):
        return 1.-((x+y)%2)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = ((2.*y)-1.).where(x==y,1.-x) * grad_output
        grad_y = ((2.*x)-1.).where(x==y,1.-y) * grad_output
        return grad_x, grad_y

def surrogate_xnor_cnf(x,y):
    return SURROGATE_XNOR.apply(x,y)

CONNECTING_FUNCTIONS={
    'ADD':add_cnf,
    'AND':and_cnf,
    'SAND':surrogate_and_cnf,
    'CSAND':const_surrogate_and_cnf,
    'IAND':iand_cnf,
    'FAST_IAND':fast_iand_cnf,
    'SIAND':surrogate_iand_cnf,
    'CSIAND':const_surrogate_iand_cnf,
    'NAND':nand_cnf,
    'FAST_NAND':fast_nand_cnf,
    'SNAND':surrogate_nand_cnf,
    'CSNAND':const_surrogate_nand_cnf,
    'OR':or_cnf,
    'FAST_OR':fast_or_cnf,
    'SOR':surrogate_or_cnf,
    'CSOR':const_surrogate_or_cnf,
    'NOR':nor_cnf,
    'FAST_NOR':fast_nor_cnf,
    'SNOR':surrogate_nor_cnf,
    'CSNOR':const_surrogate_nor_cnf,
    'RNOR':right_nor_cnf,
    'RSNOR':right_surrogate_nor_cnf,
    'SUBNOR':sub_nor_cnf,
    'SUBSNOR':sub_surrogate_nor_cnf,
    'ADDNOR':add_nor_cnf,
    'ADDSNOR':add_surrogate_nor_cnf,
    'DBNOR':double_nor_cnf,
    'DBSNOR':double_surrogate_nor_cnf,
    'LCNOR':left_constant_nor_cnf,
    'LCSNOR':left_constant_surrogate_nor_cnf,
    'CBNOR':combo_nor_cnf,
    'CBSNOR':combo_surrogate_nor_cnf,
    'HALFNOR':half_nor_cnf,
    'HALFSNOR':half_surrogate_nor_cnf,
    'LDBNOR':left_double_nor_cnf,
    'LDBSNOR':left_double_surrogate_nor_cnf,
    'BOTHHALFNOR':both_half_nor_cnf,
    'BOTHHALFSNOR':both_half_surrogate_nor_cnf,
    'XOR': xor_cnf,
    'FAST_XOR':fast_xor_cnf,
    'SXOR':surrogate_xor_cnf,
    'RXOR':right_xor_cnf,
    'RSXOR':right_surrogate_xor_cnf,
    'ADDXOR':add_xor_cnf,
    'ADDSXOR':add_surrogate_xor_cnf,
    'DBXOR':double_xor_cnf,
    'DBSXOR':double_surrogate_xor_cnf,
    'LCXOR':left_constant_xor_cnf,
    'LCSXOR':left_constant_surrogate_xor_cnf,
    'CBXOR':combo_xor_cnf,
    'CBSXOR':combo_surrogate_xor_cnf,
    'SPEXOR':special_xor_cnf,
    'BOTHHALFXOR':both_half_xor_cnf,
    'BOTHHALFSXOR':both_half_surrogate_xor_cnf,
    'XDEP':xdep_xor_cnf,
    'YDEP':ydep_xor_cnf,
    'XNOR':xnor_cnf,
    'FAST_XNOR': fast_xnor_cnf,
    'SXNOR':surrogate_xnor_cnf
}

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        cnf = cnf.upper()
        if cnf not in CONNECTING_FUNCTIONS.keys():
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
        self.cnf=CONNECTING_FUNCTIONS[cnf]
    def forward(self,x,y):
        return self.cnf(x,y)
    def __repr__(self):
        return f'ConnectingFunction({self.cnf.__name__})'