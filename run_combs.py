import itertools
import subprocess

# Define the parameter grid
batch_sizes = [32, 64, 128]
workers_list = [2, 4]
use_feature_x_options = [True, False]

# Define other fixed arguments for torchrun
nnodes = 1
nproc_per_node = 8
node_rank = 0
master_addr = '192.168.1.1'
master_port = 12355

# Generate combinations of parameters
param_combinations = list(itertools.product(batch_sizes, workers_list, use_feature_x_options))

for batch_size, workers, use_feature_x in param_combinations:
    # Construct the argument list for the script
    script_args = [
        '--batch-size', str(batch_size),
        '--workers', str(workers),
        '--log-dir', 'runs'
    ]
    if use_feature_x:
        script_args.append('--use-feature-x')

    # Construct the torchrun command
    cmd = [
        'torchrun',
        '--nnodes', str(nnodes),
        '--nproc_per_node', str(nproc_per_node),
        '--node_rank', str(node_rank),
        '--master_addr', master_addr,
        '--master_port', str(master_port),
        'train.py'
    ] + script_args

    print(f"Running command: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
