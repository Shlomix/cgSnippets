import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Cleanup the process group
def cleanup():
    dist.destroy_process_group()

# Define a deeper feedforward neural network
class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Custom communication hook
def allreduce_hook(state, bucket):
    tensors = bucket.buffer()
    dist.all_reduce(tensors, op=dist.ReduceOp.SUM)
    return tensors.div_(dist.get_world_size())

# Main training function
def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move it to the correct device
    model = DeepModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Register the custom communication hook
    ddp_model.register_comm_hook(state=None, hook=allreduce_hook)

    # Generate random data
    data = torch.randn(100, 10).to(rank)
    target = torch.randn(100, 1).to(rank)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training settings
    num_epochs = 10
    accumulation_steps = 4

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for i in range(0, len(data), accumulation_steps):
            inputs = data[i:i + accumulation_steps]
            targets = target[i:i + accumulation_steps]

            output = ddp_model(inputs)
            loss = criterion(output, targets)
            loss.backward()

            if (i // accumulation_steps + 1) % accumulation_steps == 0 or i + accumulation_steps >= len(data):
                optimizer.step()
                optimizer.zero_grad()

        if rank == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    cleanup()

# Entry point for the script
if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    train(rank, world_size)
