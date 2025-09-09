# ============================================
# FILE: mindformers/parallel_core/inference/transformer/attention.py
# ============================================

# (1) ADD THESE IMPORTS near the existing:
#     from mindspore import nn, ops as P
from mindspore.common import dtype as mstype
from mindformers.modules.flash_attention import FlashAttention
import os

# (2) INSIDE class Attention(nn.Cell).__init__(...), AFTER you read self.config
#     (right where self.use_flash_attention is set), REPLACE/WIRE LIKE THIS:
self.use_flash_attention = getattr(self.config, "use_flash_attention", False)
# Enable FA just for prefill (2D+ chunked). You can also toggle via env:
self.use_fa_for_prefill_2d = bool(os.getenv("MF_FORCE_FA_PREFILL_2D", "0") == "1") \
    or getattr(self.config, "use_fa_for_prefill_2d", False)

# (3) STILL INSIDE __init__, AFTER your q/k/v projections are constructed,
#     CREATE THE FA INSTANCE FOR PREFILL:
self._fa_prefill = None
if self.use_flash_attention:
    # scale = 1/sqrt(head_dim) is baked into FA
    self._fa_prefill = FlashAttention(
        head_num=self.num_attention_heads_per_partition,
        keep_prob=1.0,
        input_layout="BNSD",          # (B, N, S, D)
        sparse_mode=0,                # we pass an explicit causal mask
        use_attention_mask=True,
        use_actual_seqlen=True        # lets us pass ragged q/kv lens (2D+)
    )
    # Follow the same dp/mp/cp plan as attention if your wrapper supports it:
    if hasattr(self._fa_prefill, "shard"):
        self._fa_prefill.shard(self.config.parallel_config)

# (4) INSIDE Attention.construct(...), RIGHT AFTER you have q, k, v
#     (i.e., after projections + rope + reshape to (B, N, S, D)),
#     DROP THIS PREFILL BRANCH BEFORE the existing paged-attention branch:

# ---- FlashAttention for 2D+ CHUNKED PREFILL ----
is_chunked_prefill = (q_seq_lens is not None) and (self._fa_prefill is not None) \
    and self.use_flash_attention and self.use_fa_for_prefill_2d
if is_chunked_prefill:
    # 1) Always write K/V of this chunk to paged cache so decode remains paged.
    if (block_tables is not None) and (slot_mapping is not None):
        _ = self.paged_mgr(k, v, slot_mapping)

    # 2) Build a causal mask per batch/chunk for FA (shape: [B, N, Sq, Sk]).
    #    If you already receive a full mask upstream, you can reuse it:
    B = hidden_states.shape[0]
    Nq = q.shape[1]
    Sq = q.shape[2]
    Sk = k.shape[2]
    if attention_mask is None:
        tril = P.LowerTriangular()(P.Ones()((Sq, Sk), mstype.float16))  # (Sq,Sk) lower-tri
        tril = P.ExpandDims()(tril, 0)                                  # (1, Sq, Sk)
        tril = P.ExpandDims()(tril, 0)                                  # (1,1,Sq,Sk)
        attention_mask = P.Tile()(tril, (B, Nq, 1, 1))                   # (B,N,Sq,Sk)

    # 3) Handle GQA: repeat kv heads to match query heads if needed.
    if getattr(self, "use_gqa", False) and getattr(self, "n_rep", 1) > 1:
        k = P.Tile()(k, (1, self.n_rep, 1, 1))  # (B, Nk, Sk, Dh) -> repeat heads
        v = P.Tile()(v, (1, self.n_rep, 1, 1))

    # 4) Run FlashAttention (returns (B, N, Sq, Dh)).
    #    We pass q_seq_lens (chunk sizes) and batch_valid_length (prefix lens).
    fa_out = self._fa_prefill(
        query=q, key=k, value=v,
        attn_mask=attention_mask,
        actual_seq_qlen=q_seq_lens,         # 2D+ chunk lengths
        actual_seq_kvlen=batch_valid_length # running KV prefix length
    )

    # 5) Merge heads back to (B, Sq, hidden_per_partition) and exit.
    B_, N_, Sq_, Dh_ = fa_out.shape
    attn_out = fa_out.transpose(0, 2, 1, 3).reshape(B_, Sq_, N_ * Dh_)
    return self.output(attn_out)

# (5) KEEP YOUR EXISTING PAGED-ATTENTION PATH UNCHANGED BELOW.
#     That path handles both prefill (when you choose not to use FA) and decode.


# ============================================
# FILE: mindformers/models/qwen3/configuration_qwen3.py
# ============================================

# (6) IN THE Qwen3 CONFIG CLASS, DEFINE THIS NEW TOGGLE (near other bool flags):
use_flash_attention: bool = False
# Turn FA on only for 2D+ prefill; decode stays paged.
use_fa_for_prefill_2d: bool = False


# ============================================
# FILE: configs/qwen3/predict_qwen3.yaml
# ============================================

# (7) UNDER: model: -> model_config:  add the two flags:
use_flash_attention: True
use_fa_for_prefill_2d: True

# That’s it. Prefill (2D+ chunks) uses FlashAttention for the math; K/V cache writes
# and decode remain on PagedAttention. You can also toggle at runtime with:
#   export MF_FORCE_FA_PREFILL_2D=1
