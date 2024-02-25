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
        self.cupy_thresh = self.cupy_bias = self.cupy_reset = None #TODO: add support for other reset values. currently only 0 is supported

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode =='m':
            return ('torch') #TODO: fix cupy code and add back to supported backends
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
                x_seq.flatten(1),self.v.flatten(0),self.cupy_thresh.flatten(0), self.cupy_bias.flatten(0), self.v_reset, self.w.sigmoid().to(x_seq),
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








class BNLIFNode(BaseNode):
    def __init__(self, scale:torch.Tensor, bias: torch.Tensor, tau: float = 2.0, 
                 decay_input: bool = True, v_threshold: float = 1., v_reset: float = 0., 
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        #NOTE: technically some of the parameters such as the surrogate function are not required as bnlif is not used during training
        assert isinstance(tau, float) and tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        self.tau = tau
        if scale.shape==1:
            scale = scale.reshape(1, -1, 1, 1)
        self.scale = scale
        self.v_threshold = scale * v_threshold
        if v_reset is not None:
            self.v_reset = scale * v_reset
        if bias.shape==1:
            bias = bias.reshape(1, -1, 1, 1)
        self.bias = bias
        self.cupy_thresh = self.cupy_bias = self.cupy_reset = None #TODO: add support for other reset values. currently only 0 is supported

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode =='m':
            return ('torch') #TODO: add cupy code and add to supported backends
        else:
            raise ValueError(self.step_mode)
        
    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, scale={self.scale}, bias={self.bias}'
    
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

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, bias: torch.Tensor, v:torch.Tensor, tau: float):
        return v + (x + bias - v) / tau
    
    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, bias: torch.Tensor, v:torch.Tensor, tau: float, v_reset: torch.Tensor):
        return v + (x + bias - v + v_reset) / tau
    
    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, bias: torch.Tensor, v:torch.Tensor, tau: float):
        return v - v / tau + x + bias
    
    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, bias: torch.Tensor, v:torch.Tensor, tau: float, v_reset: torch.Tensor):
        return v + (v_reset - v) / tau + x + bias
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor, tau: float):
        v = v + (x + bias - v + v_reset) / tau
        spike = (v>=v_threshold).to(x)
        v = v_reset * spike + (1.-spike)*v
        return spike, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor, tau: float):
        v = v - (v - v_reset) / tau + x + bias
        spike = (v>=v_threshold).to(x)
        v = v_reset * spike + (1.-spike)*v
        return spike, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, tau: float):
        v = v + (x + bias - v) / tau
        spike = (v>=v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, tau: float):
        v = v - v / tau + x + bias
        spike = (v>=v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] + bias - (v - v_reset)) / tau
            spike = (v>=v_threshold).to(x_seq)
            v = v_reset * spike + (1.-spike)*v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t] + bias
            spike = (v>=v_threshold).to(x_seq)
            v = v_reset * spike + (1.-spike)*v
            spike_seq[t] = spike
        return spike_seq, v 
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] + bias - v) / tau
            spike = (v>=v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - v / tau + x_seq[t] + bias
            spike = (v>=v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] + bias - (v - v_reset)) / tau
            spike = (v>=v_threshold).to(x_seq)
            v = v_reset * spike + (1.-spike)*v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t] + bias
            spike = (v>=v_threshold).to(x_seq)
            v = v_reset * spike + (1.-spike)*v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] + bias - v) / tau
            spike = (v>=v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - v / tau + x_seq[t] + bias
            spike = (v>=v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                self.v_float_to_tensor(x)
                self.neuronal_charge(x)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                return spike
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.bias, self.v, self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.bias, self.v, self.v_threshold, self.tau)

            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.bias, self.v, self.v_threshold, self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.bias, self.v, self.v_threshold, self.v_reset, self.tau)

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

    def multi_step_forward(self, x_seq:torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return self._multi_step_forward(x_seq)
            elif self.backend == 'cupy':
                raise NotImplementedError(self.backend) #TODO: add cupy code and add to supported backends
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq, self.bias, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.bias, self.v, self.v_threshold, self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq, self.bias, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, self.bias, self.v, self.v_threshold, self.tau)

            else:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq, self.bias, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, self.bias, self.v, self.v_threshold, self.v_reset, self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq, self.bias, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq, self.bias, self.v, self.v_threshold, self.v_reset, self.tau)

            return spike_seq
    
    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
