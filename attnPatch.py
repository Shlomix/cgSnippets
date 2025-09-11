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
    """Forward process of the SelfAttention (instrumented)."""
    print_op = ops.Print()
    reduce_max = ops.ReduceMax(keep_dims=False)
    reduce_sum = ops.ReduceSum(keep_dims=False)
    eq = ops.Equal()
    greater = ops.Greater()

    ori_dtype = hidden_states.dtype
    bsz = hidden_states.shape[0]

    # Entry prints (very light)
    print_op("ATTN[enter] layer=", str(self.layer_number),
             " use_fa=", str(self.use_flash_attention),
             " bsz=", str(bsz),
             " hidden=", str(hidden_states.shape))

    # QKV projection
    query, key, value = self.get_query_key_value_tensors(hidden_states)
    print_op("QKV shapes q/k/v:", str(query.shape), " / ", str(key.shape), " / ", str(value.shape))

    # Rotary (kept as-is)
    if rotary_pos_cos is not None and rotary_pos_sin is not None:
        query, key = self.rotary_embedding(query, key, rotary_pos_cos, rotary_pos_sin, batch_valid_length)

    # Classification prints (decode / first prefill / 2d+ chunk)
    try:
        if q_seq_lens is not None:
            sum_q = reduce_sum(q_seq_lens)
            is_decode = bool(eq(sum_q, ops.scalar_to_tensor(bsz, sum_q.dtype)).asnumpy())
        else:
            is_decode = False

        if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None):
            max_q  = reduce_max(actual_seq_qlen)
            max_kv = reduce_max(actual_seq_kvlen)
            kv_longer_than_q = bool(greater(max_kv, max_q).asnumpy())
            same_len = bool(eq(max_kv, max_q).asnumpy())
        else:
            kv_longer_than_q = False
            same_len = False

        is_first_prefill = bool(self.is_prefill) and same_len
        is_second_plus_chunk = (not bool(self.is_prefill)) and (not is_decode) and kv_longer_than_q

        print_op("ATTN.class layer=", str(self.layer_number),
                 " is_prefill=", str(self.is_prefill),
                 " is_decode=", str(is_decode),
                 " first_prefill=", str(is_first_prefill),
                 " second_plus_chunk=", str(is_second_plus_chunk))
    except Exception as _:
        # Only best-effort prints; do not fail the graph
        pass

    # Core attention call (unchanged semantics; FA module handles chunk logic)
    if self.use_flash_attention:
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
        core_attn_out = self.core_attention(query, key, value, attention_mask)

    # Output projection
    output = self.linear_proj(core_attn_out)
    output = self.cast(output, ori_dtype)
    print_op("ATTN[exit] layer=", str(self.layer_number), " out=", str(output.shape), " dtype=", str(output.dtype))
    return output
