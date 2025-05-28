# ------------------------------------------------------------
# self_contained_chunk_prefill.py
# ------------------------------------------------------------
from dataclasses import dataclass
from typing import List, Optional
import torch

# ── ❱❱ 1. Bring in the runtime types your repo already has ❰❰ ──
#
# If the import paths differ in your project, tweak only these lines.
#
from your_pkg.kv_allocator import PageAllocator
from your_pkg.blue_types    import BlueInputData
from your_pkg.clock         import get_current_step     # same util your _prefill used


# ── ❱❱ 2. The *one* public function you asked for ❰❰ ──
#
def run_chunked_prefill_only(
    *,
    model,                      # BlueFemBoostModel (or any wrapper with .generate_tokens)
    tokenizer,                  # HF / sentencepiece tokenizer
    prompt_text: str,
    chunk_size: int = 1,        # << make it 1 for the "every token" case
    warmup_iterations: int = 0,
    page_size: int = 16,
    table_capacity: int = 2048,
    worker_table_counts: Optional[List[int]] = None,
    enable_prefix_caching: bool = False,
):
    """
    End-to-end driver that *prefill-streams* `prompt_text` in micro-batches
    of `chunk_size` tokens.

    Returns
    -------
    last_model_output : whatever `.generate_tokens()` emits for the final
                        micro-batch (all previous outputs are ignored)
    """
    # ── 2·1  Encode prompt ------------------------------------------------
    prompt_ids: List[int] = tokenizer.encode(prompt_text)
    total_tokens          = len(prompt_ids)

    # ── 2·2  Warm-up pass  (tiny prompt ignored) --------------------------
    if warmup_iterations:
        warm_ids  = tokenizer.encode("I am a warmup")
        warm_chunk= [warm_ids]                         # batch-of-1, full prompt
        for _ in range(warmup_iterations):
            _generate_prefill_once(
                model          = model,
                input_ids      = warm_chunk,
                page_mapping   = [{}],                 # empty mapping means "use next free"
                seq_lens       = [len(warm_ids)],
                context_lens   = [0],
                seq_start_loc  = [0],
                query_start_loc=[0],
                max_query_len  = len(warm_ids),
                num_input_ids  = len(warm_ids),
                enable_prefix_caching = enable_prefix_caching,
            )

    # ── 2·3  PageAllocator for the *real* run -----------------------------
    allocator = PageAllocator(
        input_ids                = [prompt_ids],          # single sequence
        worker_quota             = worker_table_counts,
        fem_boost_model          = model,
        page_size                = page_size,
        table_capacity           = table_capacity,
        preallocated_first_available_worker=None,
    )

    # ── 2·4  Stream the prompt in chunks ----------------------------------
    last_output = None
    seq_start_loc = [0]                                   # single sequence ⇒ always 0

    for idx in range(0, total_tokens, chunk_size):
        chunk          = prompt_ids[idx : idx + chunk_size]
        this_len       = len(chunk)                       # chunk_size except last
        context_len    = idx                              # tokens already parked in KV
        seq_len        = context_len + this_len           # length *after* we add the chunk

        # one-row lists = batch-size 1
        seq_lens        = [seq_len]
        context_lens    = [context_len]
        query_start_loc = [context_len]
        num_input_ids   = this_len
        max_query_len   = this_len

        # The allocator advances *one full chunk* at once
        page_mapping = allocator.step()

        # call the internal helper that mirrors _prefill’s C++ bridge
        last_output = _generate_prefill_once(
            model                = model,
            input_ids            = [chunk],
            page_mapping         = page_mapping,
            seq_lens             = seq_lens,
            context_lens         = context_lens,
            seq_start_loc        = seq_start_loc,
            query_start_loc      = query_start_loc,
            max_query_len        = max_query_len,
            num_input_ids        = num_input_ids,
            enable_prefix_caching= enable_prefix_caching,
        )

    return last_output


# ── ❱❱ 3. Tiny internal helper – *copies* your colleague’s _prefill logic ❰❰ ──
#
def _generate_prefill_once(
    *,
    model,
    input_ids,               # 2-D: [[tok₀, …]]  (batch-size == 1)
    page_mapping,            # 1-row mapping from PageAllocator
    seq_lens,
    context_lens,
    seq_start_loc,
    query_start_loc,
    max_query_len,
    num_input_ids,
    enable_prefix_caching: bool = False,
):
    """
    Build BlueInputData exactly like _prefill, but with *all* arrays
    supplied by the caller – no shape inference, no surprises.
    """
    current_step        = get_current_step()
    is_prefills         = [True]                    # single sequence
    num_prefills        = 1
    num_prefill_tokens  = num_input_ids
    num_decode_tokens   = 0
    max_prefill_seq_len = seq_lens[0]
    max_decode_seq_len  = 0

    input_data = BlueInputData(
        input_ids            = input_ids,
        current_step         = current_step,
        page_mapping         = page_mapping,
        is_prefills          = is_prefills,
        enable_prefix_caching= enable_prefix_caching,
        num_prefills         = num_prefills,
        num_prefill_tokens   = num_prefill_tokens,
        num_decode_tokens    = num_decode_tokens,
        seq_lens             = seq_lens,
        max_query_len        = max_query_len,
        max_prefill_seq_len  = max_prefill_seq_len,
        max_decode_seq_len   = max_decode_seq_len,
        query_start_loc      = query_start_loc,
        seq_start_loc        = seq_start_loc,
        context_lens         = context_lens,
        get_consumed_memory  = False,               # keep whatever default you had
    )
    # The model’s .generate_tokens() returns the decoder output for *this*
    # micro-batch.  We discard it except for the very last chunk.
    return model.generate_tokens(input_data)
