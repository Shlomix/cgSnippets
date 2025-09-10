# ==========================
# Replace: class Attention.construct(...)
# ==========================
def construct(self,
              hidden_states,
              attention_mask,
              key_value_states=None,
              rotary_pos_cos=None,
              rotary_pos_sin=None,
              position_ids=None,
              batch_valid_length=None,
              block_tables=None,
              slot_mapping=None,
              q_seq_lens=None,
              actual_seq_qlen=None,
              actual_seq_kvlen=None,
              context_lens_tensor=None,
              key_cache=None,
              value_cache=None):
    """Forward process of the SelfAttention."""
    print_op = ops.Print()

    # --- header ---
    ori_dtype = hidden_states.dtype
    print_op("ATTN.construct: layer_number=", self.layer_number)
    print_op("ATTN.flags: use_flash_attention=", str(self.use_flash_attention),
             " num_heads=", self.num_heads,
             " num_query_groups=", self.num_query_groups,
             " head_dim=", self.hidden_size_per_attention_head)

    # --- inputs presence / shape / dtype ---
    if hidden_states is None:
        print_op("IN hidden_states: None")
    else:
        print_op("IN hidden_states: shape=", str(hidden_states.shape), " dtype=", str(hidden_states.dtype))

    if attention_mask is None:
        print_op("IN attention_mask: None")
    else:
        print_op("IN attention_mask: shape=", str(attention_mask.shape), " dtype=", str(attention_mask.dtype))

    if key_value_states is None:
        print_op("IN key_value_states: None")
    else:
        print_op("IN key_value_states: shape=", str(key_value_states.shape), " dtype=", str(key_value_states.dtype))

    if rotary_pos_cos is None:
        print_op("IN rotary_pos_cos: None")
    else:
        print_op("IN rotary_pos_cos: shape=", str(rotary_pos_cos.shape), " dtype=", str(rotary_pos_cos.dtype))

    if rotary_pos_sin is None:
        print_op("IN rotary_pos_sin: None")
    else:
        print_op("IN rotary_pos_sin: shape=", str(rotary_pos_sin.shape), " dtype=", str(rotary_pos_sin.dtype))

    if position_ids is None:
        print_op("IN position_ids: None")
    else:
        print_op("IN position_ids: shape=", str(position_ids.shape), " dtype=", str(position_ids.dtype))

    if batch_valid_length is None:
        print_op("IN batch_valid_length: None")
    else:
        print_op("IN batch_valid_length: shape=", str(batch_valid_length.shape), " dtype=", str(batch_valid_length.dtype))

    if block_tables is None:
        print_op("IN block_tables: None")
    else:
        print_op("IN block_tables: shape=", str(block_tables.shape), " dtype=", str(block_tables.dtype))

    if slot_mapping is None:
        print_op("IN slot_mapping: None")
    else:
        print_op("IN slot_mapping: shape=", str(slot_mapping.shape), " dtype=", str(slot_mapping.dtype))

    if q_seq_lens is None:
        print_op("IN q_seq_lens: None")
    else:
        print_op("IN q_seq_lens: shape=", str(q_seq_lens.shape), " dtype=", str(q_seq_lens.dtype))

    if actual_seq_qlen is None:
        print_op("IN actual_seq_qlen: None")
    else:
        print_op("IN actual_seq_qlen: shape=", str(actual_seq_qlen.shape), " dtype=", str(actual_seq_qlen.dtype))

    if actual_seq_kvlen is None:
        print_op("IN actual_seq_kvlen: None")
    else:
        print_op("IN actual_seq_kvlen: shape=", str(actual_seq_kvlen.shape), " dtype=", str(actual_seq_kvlen.dtype))

    if context_lens_tensor is None:
        print_op("IN context_lens_tensor: None")
    else:
        print_op("IN context_lens_tensor: shape=", str(context_lens_tensor.shape), " dtype=", str(context_lens_tensor.dtype))

    if key_cache is None:
        print_op("IN key_cache: None")
    else:
        print_op("IN key_cache: shape=", str(key_cache.shape), " dtype=", str(key_cache.dtype))

    if value_cache is None:
        print_op("IN value_cache: None")
    else:
        print_op("IN value_cache: shape=", str(value_cache.shape), " dtype=", str(value_cache.dtype))

    # --- QKV projection ---
    query, key, value = self.get_query_key_value_tensors(hidden_states)
    print_op("QKV.after_proj query: shape=", str(query.shape), " dtype=", str(query.dtype))
    print_op("QKV.after_proj key:   shape=", str(key.shape), " dtype=", str(key.dtype))
    print_op("QKV.after_proj value: shape=", str(value.shape), " dtype=", str(value.dtype))

    # --- rotary ---
    if rotary_pos_cos is not None and rotary_pos_sin is not None:
        print_op("ROTARY: applying rotary embedding")
        query, key = self.rotary_embedding(query, key, rotary_pos_cos, rotary_pos_sin, batch_valid_length)
        print_op("ROTARY.after query: shape=", str(query.shape), " dtype=", str(query.dtype))
        print_op("ROTARY.after key:   shape=", str(key.shape), " dtype=", str(key.dtype))
    else:
        print_op("ROTARY: skipped (cos/sin is None)")

    # --- core attention call ---
    if self.use_flash_attention:
        print_op("CORE: FlashAttention path")
        core_attn_out = self.core_attention(
            query=query,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            attn_mask=attention_mask,
            key_cache=key_cache,
            value_cache=value_cache)
    else:
        print_op("CORE: non-Flash (regular) path")
        core_attn_out = self.core_attention(query, key, value, attention_mask)

    # --- output projection ---
    print_op("CORE.out: shape=", str(core_attn_out.shape), " dtype=", str(core_attn_out.dtype))
    output = self.linear_proj(core_attn_out)
    print_op("OUT.proj: shape=", str(output.shape), " dtype=", str(output.dtype))
    output = self.cast(output, ori_dtype)
    print_op("OUT.cast_to_ori_dtype: dtype=", str(output.dtype))
    return output


# ==========================
# Replace: class SelfAttention.get_query_key_value_tensors(...)
# ==========================
def get_query_key_value_tensors(self, hidden_states):
    print_op = ops.Print()
    if hidden_states is None:
        print_op("SA.get_qkv IN hidden_states: None")
    else:
        print_op("SA.get_qkv IN hidden_states: shape=", str(hidden_states.shape), " dtype=", str(hidden_states.dtype))

    qkv = self.cast(self.linear_qkv(hidden_states), self.compute_dtype)
    print_op("SA.get_qkv qkv: shape=", str(qkv.shape), " dtype=", str(qkv.dtype))

    query, key, value = mint.split(qkv,
                                   (self.hidden_size_per_partition,
                                    self.kv_hidden_size_per_partition,
                                    self.kv_hidden_size_per_partition), -1)
    print_op("SA.get_qkv split query: shape=", str(query.shape), " dtype=", str(query.dtype))
    print_op("SA.get_qkv split key:   shape=", str(key.shape), " dtype=", str(key.dtype))
    print_op("SA.get_qkv split value: shape=", str(value.shape), " dtype=", str(value.dtype))

    if self.q_layernorm is not None:
        print_op("SA.get_qkv q_layernorm: applying")
        orig_query_shape = query.shape
        query = self.q_layernorm(query.reshape(hidden_states.shape[:-1] +
                                               (-1, self.hidden_size_per_attention_head,)))
        query = query.reshape(orig_query_shape)
        print_op("SA.get_qkv q_layernorm: out shape=", str(query.shape), " dtype=", str(query.dtype))
    else:
        print_op("SA.get_qkv q_layernorm: None")

    if self.k_layernorm is not None:
        print_op("SA.get_qkv k_layernorm: applying")
        orig_key_shape = key.shape
        key = self.k_layernorm(key.reshape(hidden_states.shape[:-1] +
                                           (-1, self.hidden_size_per_attention_head,)))
        key = key.reshape(orig_key_shape)
        print_op("SA.get_qkv k_layernorm: out shape=", str(key.shape), " dtype=", str(key.dtype))
    else:
        print_op("SA.get_qkv k_layernorm: None")

    return query, key, value
