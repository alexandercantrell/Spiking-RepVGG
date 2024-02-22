import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based import surrogate
import neuron_kernel

class BNPLIFNode(BaseNode):
    def __init__(self, scale:torch.Tensor, bias: torch.Tensor, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        #NOTE: technically some of the parameters such as the surrogate function are not required as bnplif is not used during training
        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))
        if scale.shape==1:
            scale = scale.reshape(1, -1, 1, 1)
        self.scale = scale
        self.v_threshold = scale * v_threshold
        if bias.shape==1:
            bias = bias.reshape(1, -1, 1, 1)
        self.bias = bias

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode =='m':
            return ('torch',)
        else:
            raise ValueError(self.step_mode)
        
    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}, scale={self.scale}, bias={self.bias}'
    
    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0:
                self.v = self.v + (x + self.bias - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x + self.bias - self.v + self.v_reset) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0:
                self.v = self.v - self.v * self.w.sigmoid() + x + self.bias 
            else:
                self.v = self.v + (self.v_reset - self.v) * self.w.sigmoid() + x + self.bias

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def _multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        z_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            z = self.single_step_forward(x_seq[t])
            z_seq.append(z)
            if self.store_v_seq:
                v_seq.append(self.v)
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(z_seq)
    
    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return self._multi_step_forward(x_seq)
        elif self.backend=='cupy':
            hard_reset = self.v_reset is not None
            if x_seq.dtype==torch.float:
                dtype='float'
            elif x_seq.dtype==torch.half:
                dtype='half2'
            else:
                raise NotImplementedError(x_seq.dtype)
            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                       dtype=dtype,
                                                                                       decay_input=self.decay_input):
                self.forward_kernel = neuron_kernel.ParametricBNLIFNodeFPTTKernel(dtype=dtype, hard_reset=hard_reset, decay_input=self.decay_input)

            if self.cupy_thresh is None or self.cupy_thresh.shape[0] != x_seq.shape[1]:
                self.cupy_thresh = self.v_threshold.repeat(x_seq.shape[1], 1, 1, 1).to(x_seq)
            
            if self.cupy_bias is None or self.cupy_thresh.shape[0] != x_seq.shape[1]:
                self.cupy_bias = self.bias.repeat(x_seq.shape[1], 1, 1, 1).to(x_seq)

            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.ParametricBNLIFNodeATGF.apply(
                x_seq.flatten(1),self.v.flatten(0),self.cupy_thresh, self.cupy_bias, self.v_reset, self.w.sigmoid().to(x_seq),
                self.forward_kernel)
            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)
    
    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)