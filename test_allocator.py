import pytest
from lazy_page_allocator import LazyPageAllocator

# helpers ------------------------------------------------------------------
def make_sequences(n_seq: int, tok_per_seq: int):
    return [list(range(tok_per_seq)) for _ in range(n_seq)]

# suite --------------------------------------------------------------------
COMBOS = [(tp, pp, rem)
          for tp in (1, 2)
          for pp in (1, 2)
          for rem in (0, 1, 2)]

@pytest.mark.parametrize("tp,pp,rem", COMBOS)
def test_prefill_only(tp, pp, rem):
    seqs = make_sequences(4, 16)
    a = LazyPageAllocator(seqs, tp, pp, rem)
    # no decode steps
    assert a.used_remote_worker_ids() == []

@pytest.mark.parametrize("tp,pp", [(1,1),(2,1),(1,2),(2,2)])
def test_prefill_decode_one_remote(tp, pp):
    seqs = make_sequences(4, 16)
    a = LazyPageAllocator(seqs, tp, pp, remotes=1)
    for _ in range(4):                      # 4 decode steps
        a.step()
    assert a.used_remote_worker_ids() == [tp*pp]

@pytest.mark.parametrize("tp,pp", [(1,1),(2,1),(1,2),(2,2)])
def test_prefill_decode_two_remotes(tp, pp):
    seqs = make_sequences(4, 16)
    a = LazyPageAllocator(seqs, tp, pp, remotes=2)
    for _ in range(4):
        a.step()
    base = tp*pp
    assert set(a.used_remote_worker_ids()) == {base, base+1}

@pytest.mark.parametrize("tp,pp", [(1,1),(2,1),(1,2),(2,2)])
def test_capacity_forces_second_remote(tp, pp):
    seqs = make_sequences(4, 16)
    a = LazyPageAllocator(seqs, tp, pp, remotes=2)
    assert a.used_remote_worker_ids() == []   # after prefill
    for _ in range(4):
        a.step()
    assert len(a.used_remote_worker_ids()) == 2
