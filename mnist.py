import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def gather_gradients(model):
    # Gather gradients from all workers
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list = [torch.zeros_like(param.grad.data) for _ in range(dist.get_world_size())]
            dist.all_gather(grad_list, param.grad.data)
            gradients.append(grad_list)
    return gradients

def distribute_gradients(model, gradients):
    # Distribute modified gradients back to all workers
    for param, grad_list in zip(model.parameters(), gradients):
        if param.grad is not None:
            avg_grad = sum(grad_list) / dist.get_world_size()
            param.grad.data.copy_(avg_grad)

def custom_gradient_operation(gradients):
    # Example custom operation: element-wise average of gradients
    modified_gradients = []
    for grad_list in gradients:
        avg_grad = sum(grad_list) / len(grad_list)
        modified_gradients.append(avg_grad)
    return modified_gradients

def train(rank, world_size, gpu_count, processes_per_gpu):
    setup(rank, world_size)

    # Assign GPU device based on rank and processes_per_gpu
    device = (rank // processes_per_gpu) % gpu_count
    torch.cuda.set_device(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gather gradients from all workers
            gradients = gather_gradients(model)

            # Perform custom gradient operation
            modified_gradients = custom_gradient_operation(gradients)

            # Distribute modified gradients back to all workers
            distribute_gradients(model, modified_gradients)

            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def main():
    world_size = int(os.environ['WORLD_SIZE'])  # Set this environment variable to total number of processes across all nodes
    gpu_count = torch.cuda.device_count()
    processes_per_gpu = int(os.environ.get('PROCESSES_PER_GPU', 1))  # Set this to the number of processes per GPU
    rank = int(os.environ['RANK'])  # Set this environment variable to the rank of the current process
    train(rank, world_size, gpu_count, processes_per_gpu)

if __name__ == "__main__":
    main()
