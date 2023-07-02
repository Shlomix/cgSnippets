import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.multiprocessing import Process, Queue, set_start_method

def generate_text(device_id, queue, prompts):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device_id)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device_id)
        output = model.generate(**inputs, do_sample=True, max_length=50)
        result = tokenizer.decode(output[0])
        queue.put((prompt, result))

# Make sure to call this before creating any subprocesses
set_start_method('spawn')

prompts = [
    "In a world where",
    "Once upon a time",
    "The future will bring",
    "Life is like a",
    "The secret to happiness",
    # Add more prompts as needed
]

num_gpus = 2  # Set this to the number of available GPUs
queues = [Queue() for _ in range(num_gpus)]

processes = []
for i in range(num_gpus):
    # Distribute the prompts evenly among the GPUs
    gpu_prompts = prompts[i::num_gpus]
    p = Process(target=generate_text, args=(f"cuda:{i}", queues[i], gpu_prompts))
    p.start()
    processes.append(p)

# Get the results from each queue
for queue in queues:
    while not queue.empty():
        prompt, output = queue.get()
        print(f"Prompt: {prompt}")
        print(f"Generated: {output}\n")

# Wait for all processes to finish
for p in processes:
    p.join()
