import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class LargerNet(nn.Module):
    def __init__(self):
        super(LargerNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class HookA:
    def __init__(self, model):
        # Assuming some registration logic
        model.register_comm_hook(self, self.comm_hook)

    def comm_hook(self, state, bucket):
        # Custom gradient manipulation
        return bucket

    def aggregate(self):
        # Final operation before optimizer step
        pass

class HookB:
    def __init__(self, model):
        # Assuming some registration logic
        model.register_comm_hook(self, self.comm_hook)

    def comm_hook(self, state, bucket):
        # Custom gradient manipulation
        return bucket

    def aggregate(self):
        # Final operation before optimizer step
        pass

def train(rank, world_size, epochs, effective_batch_size, hook_class):
    setup(rank, world_size)
    
    # Prepare DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Setup model and optimizer
    model = LargerNet().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=1)
    hook = hook_class(ddp_model)
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    accumulation_steps = effective_batch_size // loader.batch_size
    accumulated_batches = 0

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(rank), target.cuda(rank)
            output = ddp_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            accumulated_batches += 1

            if accumulated_batches % accumulation_steps == 0:
                hook.aggregate()  # Apply final operations defined by the hook
                optimizer.step()
                optimizer.zero_grad()
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Effective Batch Size: {effective_batch_size}")

    cleanup()

def main():
    world_size = 2
    effective_batch_size = 200  # Target effective batch size
    epochs = 3
    hooks = [HookA, HookB]  # List of hook classes

    for hook_class in hooks:
        print(f"Testing with {hook_class.__name__}")
        torch.multiprocessing.spawn(train, args=(world_size, epochs, effective_batch_size, hook_class), nprocs=world_size, join=True)

if __name__ == "__main__":
   
