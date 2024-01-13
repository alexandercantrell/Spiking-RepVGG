import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based import surrogate
import neuron_kernel

class ConnIFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                backend='torch', store_v_seq: bool = False):
            super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)
        
    def neuronal_charge(self, x: torch.Tensor, y: torch.Tensor):
         self.v = self.v + x + (y * self.v_threshold)

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x + (y * v_threshold)
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v = v + x + (y * v_threshold)
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + (y_seq[t] * v_threshold)
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                          v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + (y_seq[t] * v_threshold)
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + (y_seq[t] * v_threshold)
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t] + (y_seq[t] * v_threshold)
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    def _multi_step_forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        T = x_seq.shape[0]
        z_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            z = self.single_step_forward(x_seq[t], y_seq[t])
            z_seq.append(z)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(z_seq)
    
    def multi_step_forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return self._multi_step_forward(x_seq,y_seq)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None

                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                           dtype=dtype):
                    self.forward_kernel = neuron_kernel.ConnectingIFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype):
                    self.backward_kernel = neuron_kernel.ConnectingIFNodeBPTTKernel(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = neuron_kernel.ConnectingIFNodeATGF.apply(x_seq.flatten(1), y_seq.flatten(1), self.v.flatten(0),
                                                                     self.v_threshold, self.v_reset,
                                                                     self.forward_kernel,
                                                                     self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()

                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq, y_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, y_seq, self.v, self.v_threshold)
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq, y_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold,
                                                                                                           self.v_reset)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, y_seq, self.v, self.v_threshold,
                                                                                    self.v_reset)
            return spike_seq

    def single_step_forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                self.v_float_to_tensor(x)
                self.neuronal_charge(x,y)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                return spike
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, y, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, y, self.v, self.v_threshold, self.v_reset)
            return spike

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
