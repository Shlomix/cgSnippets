import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Assuming all required imports and model definitions are already in place

class DebugDLRM_DCN(DLRM_DCN):
    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Modified forward pass to print intermediate outputs for debugging purposes.
        """
        # Dense Arch output
        embedded_dense = self.dense_arch(dense_features)
        print("DenseArch Output:", embedded_dense)

        # Sparse Arch output
        embedded_sparse = self.sparse_arch(sparse_features)
        print("SparseArch Output:", embedded_sparse)

        # Cross interaction layer (InteractionArch) output
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        print("InteractionArch Output:", concatenated_dense)

        # OverArch output (final output layer)
        logits = self.over_arch(concatenated_dense)
        print("OverArch Output:", logits)

        return logits

# Instantiate the debug model with fixed initializations as previously set up
debug_model = DebugDLRM_DCN(
    embedding_bag_collection=ebc,  # EmbeddingBagCollection instance with fixed values
    dense_in_features=dense_in_features,
    dense_arch_layer_sizes=dense_arch_layer_sizes,
    over_arch_layer_sizes=over_arch_layer_sizes,
    dcn_num_layers=dcn_num_layers,
    dcn_low_rank_dim=dcn_low_rank_dim,
    dense_device=dense_device
)

# Initialize each model layer parameter and embedding bags to fixed values as before
initialize_to_fixed_value(debug_model, fixed_value=0.5)
initialize_embedding_bags_to_fixed_value(debug_model.sparse_arch.embedding_bag_collection, fixed_value=0.1)

# Define test inputs with all ones
dense_input = torch.ones((1, dense_in_features))  # B=1, dense feature dimension
sparse_input = KeyedJaggedTensor.from_offsets_sync(
    keys=[f"f{i}" for i in range(num_sparse_features)],
    values=torch.ones((num_sparse_features), dtype=torch.long),  # All ones
    offsets=torch.tensor([0, num_sparse_features])  # Offset for single batch
)

# Run the forward pass to print layer outputs
output = debug_model(dense_input, sparse_input)
