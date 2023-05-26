import torch
import math
from spikingjelly.activation_based.auto_cuda.cfunction import constant
from spikingjelly.activation_based.surrogate import tab4_str, heaviside, SurrogateFunctionBase

def cfunction_atan_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha = constant(None, alpha, dtype)
    if dtype == 'float':
        codes = f'const float atan_backward__alpha_x = ((float) 3.14159265358979323846) * {alpha} * {x};'
        codes += f'{y} = {alpha} / (1.0f + atan_backward__alpha_x * atan_backward__alpha_x);'
        return codes

    elif dtype == 'half2':
        codes = f'const half2 atan_backward__alpha_x = __hmul2(__hmul2(__float2half2_rn((float) 3.14159265358979323846), {alpha}), {x});'
        codes += f'{y} = __h2div({alpha}, __hfma2(atan_backward__alpha_x, atan_backward__alpha_x, __float2half2_rn(1.0f)));'
        return codes

    else:
        raise NotImplementedError(dtype)

@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return torch.mul(torch.div(alpha,torch.add(torch.pow(torch.mul(x,math.pi*alpha),2),1)), grad_output), None


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.add(torch.mul(torch.atan(torch.mul(x,math.pi*alpha)),torch.div(1,math.pi)),0.5)

    @staticmethod
    def backward(grad_output, x, alpha):
        return atan_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_M_PI__alpha__x = ((float) 3.14159265358979323846) * {alpha} * {x};
            {tab4_str}const float {y} = {alpha} / (1.0f + {sg_name}_M_PI__alpha__x * {sg_name}_M_PI__alpha__x);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_M_PI__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 3.14159265358979323846), {sg_name}_alpha), {x});
            {tab4_str}const half2 {y} = __h2div({sg_name}_alpha, __hfma2({sg_name}_M_PI__alpha__x, {sg_name}_M_PI__alpha__x, __float2half2_rn(1.0f)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction_atan_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)
