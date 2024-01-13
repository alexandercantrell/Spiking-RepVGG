import torch
from spikingjelly.activation_based.auto_cuda.neuron_kernel import NeuronFPTTKernel, NeuronBPTTKernel, NeuronATGFBase
from spikingjelly.activation_based.auto_cuda import cfunction, base

class ConnectingIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, hard_reset: bool = True, dtype: str = 'float'):
        super().__init__(hard_reset, dtype)
        self.add_param(ctype=f'const {dtype} *', cname='s_seq')

    def neuronal_charge(self) -> str:
        codes = cfunction.mul(z='h_seq[t]', x='v_th', y='s_seq[t]', dtype=self.dtype)
        codes += cfunction.add(z='h_seq[t]', x='x_seq[t]', y='h_seq[t]', dtype=self.dtype)
        codes += cfunction.add(z='h_seq[t]', x='v_v_seq[t]', y='h_seq[t]', dtype=self.dtype)
        return codes

class ConnectingIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self, surrogate_function, hard_reset: bool = True, detach_reset: bool = True, dtype: str = 'float'):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.add_param(ctype=f'const {dtype} *', cname='s_seq')
        self.add_param(ctype=f'const {dtype} *', cname='grad_s_seq')

    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
    
    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
            core_codes.append(cfunction.mul(z='grad_s_seq[t]',x='v_th',y='grad_h',dtype=self.dtype))
        return super().core + '\n' + core_codes.codes
    
class ConnectingIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, s_seq:torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                forward_kernel: ConnectingIFNodeFPTTKernel, backward_kernel: ConnectingIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            's_seq': s_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel)


        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['grad_s_seq'] = torch.zeros_like(grad_spike_seq, dtype=grad_spike_seq.dtype)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        return py_dict['grad_x_seq'], py_dict['grad_s_seq'], py_dict['grad_v_init'], None, None, None, None