import math
import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional

class OneStepDelayedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, delay_steps=10, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, maximize=maximize, delay_steps=delay_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('OneStepDelayedAdam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        state['prev_grad'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq, prev_grad = state['exp_avg'], state['exp_avg_sq'], state['prev_grad']
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']

                    state['step'] += 1
                    beta1, beta2 = group['betas']

                    # Use delayed gradient after N steps
                    effective_grad = grad if state['step'] <= group['delay_steps'] else prev_grad

                    if group['weight_decay'] != 0:
                        effective_grad.add_(p.data, alpha=group['weight_decay'])

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(effective_grad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1 - beta2)

                    if group['amsgrad']:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])

                    step_size = group['lr'] / (1 - beta1 ** state['step'])

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # Update prev_grad for next iteration
                    prev_grad.copy_(grad)

        return loss
