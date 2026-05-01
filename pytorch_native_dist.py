import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import os
import json
import socket

# ── 1. DEFINE MODEL ─────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ── 2. TRAINING LOGIC ───────────────────────────────────────
def train(args):
    # Setup distributed environment if needed
    is_distributed = args.mode == 'distributed'
    rank = 0
    world_size = 1
    
    if is_distributed:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        
        # Initialize the process group
        # "gloo" is recommended for Windows/CPU training
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=args.rank,
            world_size=args.world_size
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Worker {rank}/{world_size}] Initialized on HOST: {socket.gethostname()}")
    else:
        print(f"--- STARTING LOCAL TRAINING (Single Machine) ---")

    # Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    os.makedirs('./data', exist_ok=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Data Partitioning
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
        num_samples = len(trainloader.dataset) // world_size
        print(f"[Worker {rank}] Dataset Partitioned. Processing ~{num_samples} samples this epoch.")
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print(f"Processing all {len(trainset)} samples.")

    # Initialize Model
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    
    if is_distributed:
        model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch) # Important for shuffling in distributed
            
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_end = time.time() - epoch_start
        epoch_times.append(epoch_end)
        avg_loss = running_loss / len(trainloader)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_end:.2f}s")

    total_time = time.time() - start_time
    
    # Save Results (only rank 0)
    if rank == 0:
        print(f"\n--- TRAINING COMPLETE ---")
        print(f"Total Time: {total_time:.2f}s")
        
        stats = {
            "mode": args.mode,
            "workers": world_size,
            "total_time": total_time,
            "final_loss": avg_loss,
            "epochs": args.epochs,
            "timestamp": time.ctime()
        }
        
        filename = f"results_{args.mode}_{world_size}.json"
        with open(filename, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"✅ Stats saved to {filename}")

    if is_distributed:
        dist.destroy_process_group()

# ── 3. MAIN EXECUTION ───────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Native Distributed Training Demo")
    
    # Mode Settings
    parser.add_argument('--mode', choices=['local', 'distributed'], default='local', 
                        help="Run in 'local' (single machine) or 'distributed' (multi-machine) mode.")
    
    # Distributed Settings
    parser.add_argument('--rank', type=int, default=0, help="Rank of the current process (0 is master)")
    parser.add_argument('--world_size', type=int, default=1, help="Total number of processes/nodes")
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help="IP address of the master node")
    parser.add_argument('--master_port', type=str, default='12355', help="Port for distributed communication")
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size per worker")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train(args)
