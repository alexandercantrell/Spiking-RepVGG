import math
import torch
import logging
from spikingjelly.activation_based.auto_cuda.neuron_kernel import NeuronFPTTKernel, NeuronBPTTKernel, NeuronATGFBase, if_requires_grad, scalar_to_cupy, new_tensors, neuronal_fire, neuronal_hard_reset, neuronal_soft_reset
from spikingjelly.activation_based.auto_cuda import base, cfunction
from spikingjelly.activation_based import cuda_utils
from spikingjelly import configure
from typing import Callable, Optional

try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.auto_cuda.neuronal_kernel: {e}')
    cupy = None

class ParametricConnectingLIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='s_seq')
        self.add_param(ctype=f'const {dtype} *', cname='decay')

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'
        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
        codes += cfunction.mul(z='h_seq[t]', x='v_th',y='s_seq[t]', dtype=self.dtype)
        codes += cfunction.add(z='h_seq[t]',x='h_seq[t]',y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        codes += cfunction.add(z='h_seq[t]', x='h_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

        return codes

class ParametricConnectingLIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='decay')
        self.add_param(ctype=f'float *', cname='grad_decay')
        # float to avoid overflow
        self.add_param(ctype=f'const {dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'const {dtype} *', cname='s_seq')
        self.add_param(ctype=f'{dtype} * ', cname='grad_s_seq')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay[0]', dtype=self.dtype)


    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay[0];'

    @property
    def head(self):
        # override
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        '''
        codes += fr'''
            __shared__ float sdata[{configure.cuda_threads}];
        '''
        codes += '''
            if (index < N)
            {
                const int dt = N;
        '''

        codes += self.pre_core

        if self.reverse:
            codes += '''
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            '''
        else:
            codes += '''
                for(int t = index; t < numel; t += dt)
                {
            '''
        return codes


    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        # use float to avoid overflow
        codes.append('sdata[threadIdx.x] = 0.0f;')
        return super().pre_core + '\n' + codes.codes
    
    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
            if self.decay_input:

                core_codes.append(cfunction.sub(z=f'{self.dtype} temp_var', x='h_seq[t]', y='v_v_seq[t]', dtype=self.dtype))
                core_codes.append(cfunction.mul(z='temp_var', x='temp_var', y='grad_h', dtype=self.dtype))
                core_codes.append(cfunction.div(z='temp_var', x='temp_var', y='decay[0]', dtype=self.dtype))

            else:
                if self.hard_reset:
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='v_v_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z='temp_var', x='temp_var', y='grad_h', dtype=self.dtype))
                else:
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='grad_h', y='v_v_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.neg(y='temp_var', x='temp_var', dtype=self.dtype))


            if self.dtype == 'float':
                core_codes.append('sdata[threadIdx.x] += temp_var;')
            elif self.dtype == 'half2':
                core_codes.append('sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_var), __high2half(temp_var)));')
            else:
                raise NotImplementedError(self.dtype)
        
        with base.CodeBlock(core_codes):
            core_codes.append(cfunction.mul(z='grad_s_seq[t]',x='v_th',y='grad_h',dtype=self.dtype))

        return super().core + '\n' + core_codes.codes

    @property
    def tail(self):
        codes = '''
                }
        '''

        codes += self.post_core

        codes += '''
            }
            else
            {
                sdata[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < stride)
            {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
            atomicAdd(grad_decay, sdata[0]);
            }
        }
        '''
        return codes

class ParametricConnectingLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, s_seq:torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: Optional[float], decay: torch.Tensor, forward_kernel: ParametricConnectingLIFNodeFPTTKernel, backward_kernel: ParametricConnectingLIFNodeBPTTKernel):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError('When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!')
        py_dict = {
            'x_seq': x_seq,
            's_seq': s_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)


        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], py_dict['v_v_seq'], py_dict['s_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])


        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay
        py_dict['grad_decay'] = torch.zeros_like(ctx.decay, dtype=torch.float)
        py_dict['v_v_seq'] = ctx.saved_tensors[1]
        py_dict['s_seq'] = ctx.saved_tensors[2]
        py_dict['grad_s_seq'] = torch.zeros_like(grad_spike_seq, dtype=grad_spike_seq.dtype)


        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        return py_dict['grad_x_seq'], py_dict['grad_s_seq'], py_dict['grad_v_init'], None, None,  py_dict['grad_decay'], None, None
    
class LIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='s_seq')
        self.add_param(ctype=f'const {dtype} &', cname='decay')


    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'

        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.mul(z='h_seq[t]', x='v_th',y='s_seq[t]', dtype=self.dtype)
        codes += cfunction.add(z='h_seq[t]',x='h_seq[t]',y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        codes += cfunction.add(z='h_seq[t]', x='h_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

        return codes

class LIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay;'
        
    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
            core_codes.append(cfunction.mul(z='grad_s_seq[t]',x='v_th',y='grad_h',dtype=self.dtype))
        return super().core + '\n' + core_codes.codes

class LIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, s_seq:torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: Optional[float], decay: float,
                forward_kernel: LIFNodeFPTTKernel, backward_kernel: LIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            's_seq': s_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], py_dict['s_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])


        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay
        py_dict['s_seq'] = ctx.saved_tensors[1]
        py_dict['grad_s_seq'] = torch.zeros_like(grad_spike_seq, dtype=grad_spike_seq.dtype)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')


        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None


        return py_dict['grad_x_seq'], py_dict['grad_s_seq'], py_dict['grad_v_init'], None, None, None, None, None
    


















class BatchNormNeuronFPTTKernel(base.CKernel2D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}',
            reverse=False)
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='x_seq')
        self.add_param(ctype=f'{dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'{dtype} *', cname='h_seq')
        self.add_param(ctype=f'{dtype} *', cname='spike_seq')
        self.add_param(ctype=f'{dtype} *', cname='v_th')
        self.add_param(ctype=f'{dtype} *', cname='v_bias')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    def neuronal_charge(self) -> str:
        return '// neuronal_charge should be defined here!'

    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        core_codes.append(cfunction.add(z='x_seq[t]', x='x_seq[t]', y='v_bias[t]', dtype=self.dtype))
        core_codes.append(self.neuronal_charge())

        core_codes.append(neuronal_fire(spike='spike_seq[t]', v='h_seq[t]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_reset='v_reset',
                                    dtype=self.dtype))
        else:
            core_codes.append(
                neuronal_soft_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_th='v_th',
                                    dtype=self.dtype))

        self._core = core_codes.codes
        return self._core

class BatchNormNeuronATGFBase:
    @staticmethod
    def pre_forward(py_dict: dict):
        device = py_dict['x_seq'].get_device()
        scalar_to_cupy(py_dict)

        new_tensors(('h_seq', 'spike_seq', 'v_seq'), py_dict)
        py_dict['v_v_seq'] = torch.cat((py_dict.pop('v_init').unsqueeze(0), py_dict.pop('v_seq')))
        numel = py_dict['x_seq'].numel()
        N = py_dict['x_seq'].shape[1]
        threads = configure.cuda_threads
        if py_dict['x_seq'].dtype == torch.float16:
            # we will take two neurons to calculate as one neuron in cuda half2
            # pad will be implemented by the kernel.__call__
            N = math.ceil(N / 2)
            numel = N * py_dict['x_seq'].shape[0]

        blocks = cuda_utils.cal_blocks(N)

        with cuda_utils.DeviceEnvironment(device):
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)

        py_dict['numel'] = numel
        py_dict['N'] = N

        return blocks, threads, py_dict

class ParametricBNLIFNodeFPTTKernel(BatchNormNeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='decay')

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'
        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='v_v_seq[t]', dtype=self.dtype)

        return codes

class ParametricBNLIFNodeATGF(BatchNormNeuronATGFBase):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: torch.Tensor, v_bias: torch.Tensor, v_reset: Optional[float], decay: torch.Tensor, forward_kernel: ParametricBNLIFNodeFPTTKernel):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError('When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!')
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_bias': v_bias,
            'v_reset': v_reset,
            'decay': decay,
        }
        blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)


        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]