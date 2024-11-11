# Assuming all required imports and model definitions are already in place
import torch

# Define dense input with all ones
dense_input = torch.ones((1, dense_in_features))  # B=1

# Initialize each model layer parameter to a fixed value of 0.5
def initialize_to_fixed_value(module, fixed_value=0.5):
    if hasattr(module, 'weight') and module.weight is not None:
        torch.nn.init.constant_(module.weight, fixed_value)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias, fixed_value)

model.apply(lambda m: initialize_to_fixed_value(m, fixed_value=0.5))

# Initialize embedding tables to fixed value 0.1
def initialize_embedding_bags_to_fixed_value(embedding_bag_collection, fixed_value=0.1):
    for embedding_bag in embedding_bag_collection.embedding_bags.values():
        torch.nn.init.constant_(embedding_bag.weight, fixed_value)

initialize_embedding_bags_to_fixed_value(model.sparse_arch.embedding_bag_collection)

# Example sparse input (all keys and features set to 1)
sparse_input = KeyedJaggedTensor.from_offsets_sync(
    keys=[f"f{i}" for i in range(num_sparse_features)],
    values=torch.ones((1, num_sparse_features), dtype=torch.long),
    offsets=torch.tensor([0, num_sparse_features])
)

# Forward pass with fixed inputs to check layer outputs
output = model(dense_input, sparse_input)
