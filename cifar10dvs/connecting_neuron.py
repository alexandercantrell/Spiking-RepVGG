import math
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based import surrogate
import neuron_kernel


class ParaConnLIFNode(BaseNode):
    
    '''
    Parametric Connecting Leaky Integrate-and-Fire (LIF) Neuron:
    This neuron moves the connecting function from outside to inside the neuron. In this way, we are able to 
    circumvent the problem of applying Spiking Resnet to Parametric Leaky Integrate-and-Fire (LIF) Neurons. 
    The formulation of this neuron is as follows:
        v(t+1) = v(t) + (x(t) - v(t)) * w(t) + s(t) * v(thres)
        o(t) = 1 if v(t) >= v(thres) else 0
        v(t+1) = v(reset) if o(t) == 1 else v(t+1)
    Where v(t) is the membrane potential at time t, v(thres) is the threshold potential, v(reset) is the reset
    potential, s(t) is the spike of the previous layer at time t, x(t) is the input at time t, w(t) is the 
    weight at time t, and o(t) is the output spike at time t. The weight w(t) is a learnable parameter.
    '''
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def y_multiplier(self):
        if self.decay_input:
            return self.v_threshold/self.w.sigmoid()
        else:
            return self.v_threshold

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor, y: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0:
                self.v = self.v + (x - self.v) * self.w.sigmoid() + y * self.v_threshold
            else:
                self.v = self.v + (x - self.v + self.v_reset) * self.w.sigmoid() + y * self.v_threshold
        else:
            if self.v_reset is None or self.v_reset == 0:
                self.v = self.v - self.v * self.w.sigmoid() + x + y * self.v_threshold
            else:
                self.v = self.v + (self.v_reset - self.v) * self.w.sigmoid() + x + y * self.v_threshold

    def single_step_forward(self, x: torch.Tensor, y: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x,y)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

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
        if self.backend == 'torch':
            return self._multi_step_forward(x_seq, y_seq)
        elif self.backend == 'cupy':
            hard_reset = self.v_reset is not None 
            if x_seq.dtype == torch.float:
                dtype='float'
            elif x_seq.dtype == torch.half:
                dtype='half2'
            else:
                raise NotImplementedError(x_seq.dtype)  
            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                       dtype=dtype,
                                                                                       decay_input=self.decay_input):
                self.forward_kernel = neuron_kernel.ParametricConnectingLIFNodeFPTTKernel(decay_input=self.decay_input,
                                                                                   hard_reset=hard_reset, dtype=dtype)
            if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                self.backward_kernel = neuron_kernel.ParametricConnectingLIFNodeBPTTKernel(decay_input=self.decay_input,
                                                                                    surrogate_function=self.surrogate_function.cuda_codes,
                                                                                    hard_reset=hard_reset,
                                                                                    detach_reset=self.detach_reset,
                                                                                    dtype=dtype)
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.ParametricConnectingLIFNodeATGF.apply(
                x_seq.flatten(1),y_seq.flatten(1),self.v.flatten(0),self.v_threshold,self.v_reset,self.w.sigmoid().to(x_seq),
                self.forward_kernel, self.backward_kernel)
            
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


class ConnLIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.tau = tau
        self.decay_input = decay_input

    @property
    def y_multiplier(self):
        if self.decay_input:
            return self.v_threshold * self.tau
        else:
            return self.v_threshold

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor, y: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, y, self.v, self.tau, self.v_threshold)
            else:
                self.v = self.neuronal_charge_decay_input(x, y, self.v, self.v_reset, self.tau, self.v_threshold)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, y, self.v, self.tau, self.v_threshold)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, y, self.v, self.v_reset, self.tau, self.v_threshold)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, tau: float, v_threshold: float):
        v = v + (x - v) / tau + y * v_threshold
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, v_threshold: float):
        v = v + (x - (v - v_reset)) / tau + y * v_threshold
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, tau: float, v_threshold: float):
        v = v * (1. - 1. / tau) + x + y * v_threshold
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, v_threshold: float):
        v = v - (v - v_reset) / tau + x + y * v_threshold
        return v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, y:torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau + y * v_threshold
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, y:torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x + y * v_threshold
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, y:torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau + y * v_threshold
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, y:torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x + y * v_threshold
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, y_seq:torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, y_seq:torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, y_seq:torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t] + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, y_seq:torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float, v_reset: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t] + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t] + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, y_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t] + y_seq[t] * v_threshold
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                self.v_float_to_tensor(x)
                self.neuronal_charge(x,y)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                return spike
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, y, self.v,
                                                                                             self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, y, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, y, self.v,
                                                                                             self.v_threshold,
                                                                                             self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, y, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
            return spike

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
                return self._multi_step_forward(x_seq, y_seq)
            elif self.backend == 'cupy':

                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                           dtype=dtype,
                                                                                           decay_input=self.decay_input):
                    self.forward_kernel = neuron_kernel.ConnectingLIFNodeFPTTKernel(decay_input=self.decay_input,
                                                                             hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                    self.backward_kernel = neuron_kernel.ConnectingLIFNodeBPTTKernel(decay_input=self.decay_input,
                                                                              surrogate_function=self.surrogate_function.cuda_codes,
                                                                              hard_reset=hard_reset,
                                                                              detach_reset=self.detach_reset,
                                                                              dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = neuron_kernel.ConnectingLIFNodeATGF.apply(x_seq.flatten(1), y_seq.flatten(1), self.v.flatten(0),
                                                                      self.v_threshold, self.v_reset, 1. / self.tau,
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
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                            x_seq, y_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, y_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
                            x_seq, y_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, y_seq, self.v,
                                                                                                       self.v_threshold,
                                                                                                       self.tau)
            else:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
                            x_seq, y_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, y_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.v_reset,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(
                            x_seq, y_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq, y_seq, self.v,
                                                                                                       self.v_threshold,
                                                                                                       self.v_reset,
                                                                                                       self.tau)

            return spike_seq