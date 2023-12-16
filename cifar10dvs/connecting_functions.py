import torch
from torch import nn 
from torch.autograd import Function

def add_cnf(x,y):
    '''
    Combine two spike tensors by an ADD gate.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  1  1
    1 0 1  1  1
    1 1 2  1  1
    
    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return x+y

def and_cnf(x,y):
    '''
    Combine two spike tensors by an AND gate.

    Combinations
    ------------
    x y z dx dy
    0 0 0  0  0
    0 1 0  1  0
    1 0 0  0  1
    1 1 1  1  1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return x*y

class SURROGATE_AND(Function):
    '''
    Combine two spike tensors by an AND gate. Backpropagation is done by a 
    surrogate function, fixing the issue of both gradients being zero for 
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 0  1  0
    1 0 0  0  1
    1 1 1  1  1
    '''
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
    '''
    Combine two spike tensors by an AND gate. Backpropagation is done by a 
    surrogate function, fixing the issue of both gradients being zero for 
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 0  1  0
    1 0 0  0  1
    1 1 1  1  1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return SURROGATE_AND.apply(x,y)

def iand_cnf(x,y):
    '''
    Combine two spike tensors by an IAND gate.

    Combinations
    ------------
    x y z dx dy
    0 0 0  0  1
    0 1 1 -1  1
    1 0 0  0  0
    1 1 0 -1  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return (1.-x) * y

class IAND(Function):
    '''
    Combine two spike tensors by an IAND gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------ 
    x y z dx dy
    0 0 0  0  1
    0 1 1 -1  1
    1 0 0  0  0
    1 1 0 -1  0
    '''
    @staticmethod
    def forward(x,y):
        return nn.functional.relu(y-x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = -y*grad_output
        grad_y = (1.-x) * grad_output
        return grad_x, grad_y

def fast_iand_cnf(x,y):
    '''
    Combine two spike tensors by an IAND gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------ 
    x y z dx dy
    0 0 0  0  1
    0 1 1 -1  1
    1 0 0  0  0
    1 1 0 -1  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return IAND.apply(x,y)

class SURROGATE_IAND(Function):
    '''
    Combine two spike tensors by an IAND gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 0  0  1
    0 1 1 -1  1
    1 0 0 -1  1
    1 1 0 -1  0
    '''
    @staticmethod
    def forward(x,y):
        return nn.functional.relu(y-x)
    @staticmethod
    def setup_context(ctx,inputs,output):
        x,y = inputs
        ctx.save_for_backward(x,y)
    @staticmethod
    def backward(ctx,grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        grad_x = -y.where(x==y,1.) * grad_output
        grad_y = (1-x).where(x==y,1.) * grad_output
        return grad_x, grad_y
    
def surrogate_iand_cnf(x,y):
    '''
    Combine two spike tensors by an IAND gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 0  0  1
    0 1 1 -1  1
    1 0 0 -1  1
    1 1 0 -1  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor   
        Combined spike output
    '''
    return SURROGATE_IAND.apply(x,y)
    
def nand_cnf(x,y):
    '''
    Combine two spike tensors by a NAND gate.

    Combinations
    ------------
    x y z dx dy
    0 0 1  0  0
    0 1 1 -1  0
    1 0 1  0 -1
    1 1 0 -1 -1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return 1.-x*y

class SURROGATE_NAND(Function):
    '''
    Combine two spike tensors by a NAND gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 1 -1  0
    1 0 1  0 -1
    1 1 0 -1 -1
    '''
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
        grad_x = -y.where(x!=y,1.) * grad_output
        grad_y = -x.where(x!=y,1.) * grad_output
        return grad_x, grad_y
    
def surrogate_nand_cnf(x,y):
    '''
    Combine two spike tensors by a NAND gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 1 -1  0
    1 0 1  0 -1
    1 1 0 -1 -1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor   
        Combined spike output
    '''
    return SURROGATE_NAND.apply(x,y)

def or_cnf(x,y):
    '''
    Combine two spike tensors by an OR gate.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 1  0  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return x+y-x*y

class OR(Function):
    '''
    Combine two spike tensors by an OR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 1  0  0
    '''
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
    '''
    Combine two spike tensors by an OR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 1  0  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return OR.apply(x,y)

class SURROGATE_OR(Function):
    '''
    Combine two spike tensors by an OR gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 1  1  1
    '''
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
    '''
    Combine two spike tensors by an OR gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 1  1  1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return SURROGATE_OR.apply(x,y)

def nor_cnf(x,y):
    '''
    Combine two spike tensors by a NOR gate.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 0  0  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return 1.-(x+y-x*y)

class NOR(Function):
    '''
    Combine two spike tensors by a NOR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 0  0  0
    '''
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
    '''
    Combine two spike tensors by a NOR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 0  0  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return NOR.apply(x,y)

class SURROGATE_NOR(Function):
    '''
    Combine two spike tensors by a NOR gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 0 -1 -1
    '''
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
    '''
    Combine two spike tensors by a NOR gate. Backpropagation is done by a
    surrogate function, fixing the issue of both gradients being zero for
    certain value combinations.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 0 -1 -1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return SURROGATE_NOR.apply(x,y)

def xor_cnf(x,y):
    '''
    Combine two spike tensors by an XOR gate.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1 -1  1
    1 0 1  1 -1
    1 1 0 -1 -1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return x+y-(2.*x*y)

class XOR(Function):
    '''
    Combine two spike tensors by an XOR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1 -1  1
    1 0 1  1 -1
    1 1 0 -1 -1
    '''
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
    '''
    Combine two spike tensors by an XOR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1 -1  1
    1 0 1  1 -1
    1 1 0 -1 -1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return XOR.apply(x,y)
    
class SURROGATE_XOR(Function):
    '''
    Combine two spike tensors by an XOR gate. Backpropagation is done by a
    surrogate function, decreasing the frequency of input spikes and fixing
    potential issues with gradients going in opposite directions.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 0 -1 -1
    '''
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
    '''
    Combine two spike tensors by an XOR gate. Backpropagation is done by a
    surrogate function, decreasing the frequency of input spikes and fixing
    potential issues with gradients going in opposite directions.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  1
    0 1 1  0  1
    1 0 1  1  0
    1 1 0 -1 -1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return SURROGATE_XOR.apply(x,y)

class FOCUS_X_XOR(Function):
    '''
    Combine two spike tensors by an XOR gate. Backpropagation is done by a
    surrogate function which focuses on the x input, only propagating gradients
    to the y input when it is 1 and the x input is 0.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  0
    0 1 1  0  1
    1 0 1  1  0
    1 1 0 -1  0
    '''
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
        grad_y = y.where(x!=y,0) * grad_output
        grad_x = (1.-(2.*y)).where(grad_y==0,0) * grad_output
        return grad_x, grad_y
    
def focus_x_xor_cnf(x,y):
    '''
    Combine two spike tensors by an XOR gate. Backpropagation is done by a
    surrogate function which focuses on the x input, only propagating gradients
    to the y input when it is 1 and the x input is 0.

    Combinations
    ------------
    x y z dx dy
    0 0 0  1  0
    0 1 1  0  1
    1 0 1  1  0
    1 1 0 -1  0

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return FOCUS_X_XOR.apply(x,y)

def xnor_cnf(x,y):
    '''
    Combine two spike tensors by an XNOR gate.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  1 -1
    1 0 0 -1  1
    1 1 1  1  1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return 1.-(x+y-(2.*x*y))

class XNOR(Function):
    '''
    Combine two spike tensors by an XNOR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  1 -1
    1 0 0 -1  1
    1 1 1  1  1
    '''
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
    '''
    Combine two spike tensors by an XNOR gate in a more efficient way
    for forward and back propagation.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  1 -1
    1 0 0 -1  1
    1 1 1  1  1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return XNOR.apply(x,y)

class SURROGATE_XNOR(Function):
    '''
    Combine two spike tensors by an XNOR gate. Backpropagation is done by a
    surrogate function, decreasing the frequency of input spikes and fixing
    potential issues with gradients going in opposite directions.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 1  1  1
    '''
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
        grad_x = ((2.*y)-1.).where(x==y,-x) * grad_output
        grad_y = ((2.*x)-1.).where(x==y,-y) * grad_output
        return grad_x, grad_y

def surrogate_xnor_cnf(x,y):
    '''
    Combine two spike tensors by an XNOR gate. Backpropagation is done by a
    surrogate function, decreasing the frequency of input spikes and fixing
    potential issues with gradients going in opposite directions.

    Combinations
    ------------
    x y z dx dy
    0 0 1 -1 -1
    0 1 0  0 -1
    1 0 0 -1  0
    1 1 1  1  1

    Parameters
    ----------
    x : torch.Tensor
        Current spike output
    y : torch.Tensor
        Previous spike output

    Returns
    -------
    torch.Tensor
        Combined spike output
    '''
    return SURROGATE_XNOR.apply(x,y)

CONNECTING_FUNCTIONS={
    #ADD functions
    'ADD':add_cnf,
    #AND functions
    'AND':and_cnf,
    'SAND':surrogate_and_cnf,
    #IAND functions
    'IAND':iand_cnf,
    'FAST_IAND':fast_iand_cnf,
    'SIAND':surrogate_iand_cnf,
    #NAND functions
    'NAND':nand_cnf,
    'SNAND':surrogate_nand_cnf,
    #OR functions
    'OR':or_cnf,
    'FAST_OR':fast_or_cnf,
    'SOR':surrogate_or_cnf,
    #NOR functions
    'NOR':nor_cnf,
    'FAST_NOR':fast_nor_cnf,
    'SNOR':surrogate_nor_cnf,
    #XOR functions
    'XOR': xor_cnf,
    'FAST_XOR':fast_xor_cnf,
    'SXOR':surrogate_xor_cnf,
    'FXXOR':focus_x_xor_cnf,
    #XNOR functions
    'XNOR':xnor_cnf,
    'FAST_XNOR': fast_xnor_cnf,
    'SXNOR':surrogate_xnor_cnf,
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