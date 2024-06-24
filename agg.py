import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Create a simple dataset
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10)


def original_algorithm(all_grads):
    # Example reduction: mean of gradients
    mean_grads = [torch.mean(torch.stack(grad_list), dim=0) for grad_list in zip(*all_grads)]
    return mean_grads


def collect_and_reduce_gradients(model, dataloader, criterion, effective_batch_size):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    local_grads = []
    all_grads = []

    for i, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        local_grads.append([param.grad.clone() for param in model.parameters()])

        if (i + 1) % effective_batch_size == 0:
            all_grads.append(local_grads)
            local_grads = []

            # Gather gradients from all workers
            gathered_grads = [torch.zeros_like(param.grad) for param in model.parameters()]
            for grads in all_grads:
                for j, param_grads in enumerate(grads):
                    torch.distributed.all_gather(gathered_grads, param_grads[j])

            # Perform the original gradient reduction algorithm
            reduced_grads = original_algorithm(gathered_grads)
            
            # Apply reduced gradients
            with torch.no_grad():
                for param, reduced_grad in zip(model.parameters(), reduced_grads):
                    param.grad = reduced_grad
            
            optimizer.step()

            all_grads = []

    return model


class CustomDDP(DDP):
    def __init__(self, module, effective_batch_size, *args, **kwargs):
        super(CustomDDP, self).__init__(module, *args, **kwargs)
        self.effective_batch_size = effective_batch_size
        self.local_grads = []

    def reduce_gradients(self):
        # Example reduction: allreduce to sum gradients across workers
        for param in self.module.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

    def step(self, *args, **kwargs):
        self.reduce_gradients()
        super().step(*args, **kwargs)


def setup_ddp(rank, world_size, effective_batch_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    model = SimpleModel().to(rank)
    model = CustomDDP(model, effective_batch_size, device_ids=[rank])
    return model


def cleanup_ddp():
    dist.destroy_process_group()


def ddp_training_step(rank, world_size, model, dataloader, criterion):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for i, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        model.local_grads.append([param.grad.clone() for param in model.parameters()])

        if (i + 1) % model.effective_batch_size == 0:
            for grads in model.local_grads:
                for param, grad in zip(model.parameters(), grads):
                    param.grad += grad
            model.local_grads = []

            model.reduce_gradients()

            optimizer.step()


def compare_outputs(rank, world_size, effective_batch_size):
    model = SimpleModel()
    criterion = nn.MSELoss()
    
    # Standalone version
    standalone_model = model
    standalone_model = collect_and_reduce_gradients(standalone_model, dataloader, criterion, effective_batch_size)
    
    # DDP version
    setup_ddp(rank, world_size, effective_batch_size)
    ddp_model = setup_ddp(rank, world_size, effective_batch_size)
    ddp_training_step(rank, world_size, ddp_model, dataloader, criterion)
    
    # Compare parameters
    for standalone_param, ddp_param in zip(standalone_model.parameters(), ddp_model.module.parameters()):
        assert torch.allclose(standalone_param, ddp_param), "Parameters do not match!"
    
    cleanup_ddp()


def main():
    world_size = 2
    effective_batch_size = 2  # Example effective batch size
    mp.spawn(compare_outputs, args=(world_size, effective_batch_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
