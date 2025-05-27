def stream_one_token_prefill(model, prompt_ids, allocator, blue_args):
    """
    Send ONE token per _prefill() call.
    The final generate_tokens() output is returned; earlier ones are ignored.

    Parameters
    ----------
    model       : your BlueFemBoostModel (or plain LLM wrapper)
    prompt_ids  : list[int]  *already* BPE-encoded prompt
    allocator   : PageAllocator (advances KV-tables one chunk at a time)
    blue_args   : argparse.Namespace – reused by _prefill() unchanged
    """
    last_output = None
    seq_start_loc = [0]                  # never changes for a single row

    for idx, tok in enumerate(prompt_ids):
        # ---- build the 1-token “micro-batch” bookkeeping ------------
        chunk            = [[tok]]       # 2-D: batch=1, len=1
        seq_lens         = [idx + 1]
        context_lens     = [idx]
        query_start_loc  = [idx]
        num_input_ids    = 1
        max_query_len    = 1

        # page-mapping advances one step per chunk
        page_mapping = allocator.step()          # shape (1, ·)

        # hand everything to your existing helper
        last_output = _prefill(
            model,
            chunk,                     # input_ids
            page_mapping,
            seq_lens          = seq_lens,
            context_lens      = context_lens,
            seq_start_loc     = seq_start_loc,
            query_start_loc   = query_start_loc,
            max_query_len     = max_query_len,
            num_input_ids     = num_input_ids,
        )
        # ignore `last_output` until the very end

    return last_output                    # matters only after the last token
