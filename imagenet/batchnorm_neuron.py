from typing import Callable
import torch
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based import surrogate

class BNIFNode(BaseNode):
    def __init__(self, scale:torch.Tensor, bias: torch.Tensor, v_threshold: float = 1., v_reset: float = 0.,
                surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                backend='torch', store_v_seq: bool = False):
            #NOTE: technically some of the parameters such as the surrogate function are not required as bnif is not used during training
            super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
            if scale.shape==1:
                scale = scale.reshape(1,-1,1,1)
            self.scale = scale
            self.v_threshold = scale * v_threshold
            if v_reset is not None:
                self.v_reset = scale * v_reset
            if bias.shape==1:
                bias = bias.reshape(1,-1,1,1)
            self.bias = bias
            self.cupy_thresh=self.cupy_bias=self.cupy_reset=None #TODO: add cupy support

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch')
        elif self.step_mode == 'm':
            return ('torch') #TODO: add cupy support
        else:
            raise ValueError(self.step_mode)
        
    def neuronal_charge(self, x: torch.Tensor):
         self.v = self.v + x + self.bias

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor, v_reset: torch.Tensor):
        v = v + x + bias
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor):
        v = v + x + bias
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor,
                                               v_reset: torch.Tensor):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + bias
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor,
                                                          v_reset: torch.Tensor):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + bias
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + bias
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, bias: torch.Tensor, v: torch.Tensor, v_threshold: torch.Tensor):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + bias
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
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
        if self.training:
            if self.backend == 'torch':
                return self._multi_step_forward(x_seq)
            elif self.backend == 'cupy':
                raise NotImplementedError(self.backend) #TODO: add cupy code and add to supported backends
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq, self.bias,
                                                                                                           self.v,
                                                                                                           self.v_threshold)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.bias, self.v, self.v_threshold)
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq, self.bias,
                                                                                                           self.v,
                                                                                                           self.v_threshold,
                                                                                                           self.v_reset)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.bias, self.v, self.v_threshold,
                                                                                    self.v_reset)
            return spike_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                self.v_float_to_tensor(x)
                self.neuronal_charge(x)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                return spike
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.bias, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.bias, self.v, self.v_threshold, self.v_reset)
            return spike

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
