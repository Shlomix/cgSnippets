import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
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
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)  # Ensure consistent initialization

def cleanup():
    dist.destroy_process_group()

def gather_gradients(model):
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list = [torch.zeros_like(param.grad.data) for _ in range(dist.get_world_size())]
            dist.all_gather(grad_list, param.grad.data)
            gradients.append(grad_list)
    return gradients

def distribute_gradients(model, modified_gradients):
    for param, new_grad in zip(model.parameters(), modified_gradients):
        if param.grad is not None:
            param.grad.data.copy_(new_grad)

def custom_gradient_operation(gradients):
    modified_gradients = []
    for grad_list in gradients:
        avg_grad = sum(grad_list) / len(grad_list)
        modified_gradients.append(avg_grad)
    return modified_gradients

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def train(rank, world_size, gpu_count, global_batch_size):
    setup(rank, world_size)

    # Ensure global_batch_size is divisible by world_size and greater than world_size
    assert global_batch_size % world_size == 0, "Global batch size must be divisible by the number of workers"
    assert global_batch_size >= world_size, "Global batch size must be greater than or equal to the number of workers"
    
    local_batch_size = global_batch_size // world_size

    # Assign GPU device based on rank
    device = rank % gpu_count
    torch.cuda.set_device(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=local_batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    model = Net().to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Synchronize to ensure all processes start at the same point
    dist.barrier()

    for epoch in range(10):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gather gradients from all workers
            gradients = gather_gradients(ddp_model)

            # Perform custom gradient operation
            modified_gradients = custom_gradient_operation(gradients)

            # Distribute modified gradients back to all workers
            distribute_gradients(ddp_model, modified_gradients)

            optimizer.step()

            # Synchronize to ensure all processes have the same state
            dist.barrier()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")
        
        # Evaluate the model
        if rank == 0:  # Only the master process should print
            evaluate(ddp_model.module, device, test_loader)

    cleanup()

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST DDP Example')
    parser.add_argument('--batch-size', type=int, required=True, help='global batch size for training')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers (default: 1)')
    args = parser.parse_args()

    world_size = int(os.getenv('WORLD_SIZE'))
    rank = int(os.getenv('RANK'))
    gpu_count = torch.cuda.device_count()

    train(rank, world_size, gpu_count, args.batch_size)

if __name__ == "__main__":
    main()
