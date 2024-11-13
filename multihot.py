from typing import List, Tuple
import numpy as np
import torch
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class Multihot:
    def __init__(
        self,
        multi_hot_sizes: List[int],
        num_embeddings_per_feature: List[int],
        batch_size: int,
        collect_freqs_stats: bool,
        dist_type: str = "uniform",
        force_value: int = None  # Optional parameter to set a fixed multi-hot value
    ):
        if dist_type not in {"uniform", "pareto"}:
            raise ValueError(
                f"Multi-hot distribution type {dist_type} is not supported. "
                'Only "uniform" and "pareto" are supported.'
            )
        self.dist_type = dist_type
        self.multi_hot_sizes = multi_hot_sizes
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.batch_size = batch_size
        self.force_value = force_value  # Store the forced value

        # Generate multi-hot tables
        self.multi_hot_tables_l = self.__make_multi_hot_indices_tables(
            dist_type, multi_hot_sizes, num_embeddings_per_feature
        )

        # Precompute offsets for pooling
        self.offsets = self.__make_offsets(multi_hot_sizes, batch_size)

        # Frequency stats setup
        self.collect_freqs_stats = collect_freqs_stats
        self.model_to_track = None
        self.freqs_pre_hash = []
        self.freqs_post_hash = []
        for embs_count in num_embeddings_per_feature:
            self.freqs_pre_hash.append(np.zeros((embs_count)))
            self.freqs_post_hash.append(np.zeros((embs_count)))

    def __make_multi_hot_indices_tables(
        self,
        dist_type: str,
        multi_hot_sizes: List[int],
        num_embeddings_per_feature: List[int],
    ) -> List[torch.Tensor]:
        """
        Creates multi-hot lookup tables. If `force_value` is set, all entries are set to `force_value`.
        Otherwise, values are generated based on `dist_type`.
        """
        np.random.seed(0)
        multi_hot_tables_l = []
        for embs_count, multi_hot_size in zip(num_embeddings_per_feature, multi_hot_sizes):
            if self.force_value is not None:
                # If force_value is set, create tables filled with that value
                multi_hot_table = np.full((embs_count, multi_hot_size), self.force_value)
            else:
                # Default behavior: Generate multi-hot tables based on distribution type
                embedding_ids = np.arange(embs_count)[:, np.newaxis]
                if dist_type == "uniform":
                    synthetic_sparse_ids = np.random.randint(
                        0, embs_count, size=(embs_count, multi_hot_size - 1)
                    )
                elif dist_type == "pareto":
                    synthetic_sparse_ids = (
                        np.random.pareto(a=0.25, size=(embs_count, multi_hot_size - 1))
                        .astype(np.int32) % embs_count
                    )
                multi_hot_table = np.concatenate((embedding_ids, synthetic_sparse_ids), axis=-1)
            multi_hot_tables_l.append(torch.from_numpy(multi_hot_table).int())
        return multi_hot_tables_l

    def __make_offsets(
        self,
        multi_hot_sizes: List[int],
        batch_size: int,
    ) -> torch.Tensor:
        lS_o = torch.ones((len(multi_hot_sizes) * batch_size), dtype=torch.int32)
        for k, multi_hot_size in enumerate(multi_hot_sizes):
            lS_o[k * batch_size : (k + 1) * batch_size] = multi_hot_size
        lS_o = torch.cumsum(torch.cat((torch.tensor([0]), lS_o)), dim=0)
        return lS_o

    def __make_new_batch(
        self,
        lS_i: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lS_i = lS_i.reshape(-1, batch_size)
        multi_hot_ids_l = []
        for k, (sparse_data_batch_for_table, multi_hot_table) in enumerate(
            zip(lS_i, self.multi_hot_tables_l)
        ):
            # Use the fixed value if force_value is set; otherwise, apply the embedding lookup
            if self.force_value is not None:
                # Directly fill the batch with `force_value` if set
                multi_hot_ids = torch.full_like(sparse_data_batch_for_table, self.force_value)
            else:
                # Default behavior: embedding lookup for random values
                multi_hot_ids = torch.nn.functional.embedding(sparse_data_batch_for_table, multi_hot_table)
            multi_hot_ids_l.append(multi_hot_ids.reshape(-1))
        lS_i = torch.cat(multi_hot_ids_l)
        return lS_i, self.offsets

    def convert_to_multi_hot(self, batch: Batch) -> Batch:
        batch_size = len(batch.dense_features)
        lS_i = batch.sparse_features._values
        lS_i, lS_o = self.__make_new_batch(lS_i, batch_size)
        new_sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=batch.sparse_features._keys,
            values=lS_i,
            offsets=lS_o,
        )
        return Batch(
            dense_features=batch.dense_features,
            sparse_features=new_sparse_features,
            labels=batch.labels,
        )
