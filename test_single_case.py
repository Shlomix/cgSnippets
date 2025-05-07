import json, os, pytest
from lazy_allocator import LazyBatchAllocator, DummyBoostModel

def pytest_addoption(parser):
    parser.addoption("--case", help="path to JSON case (or set CASE_PATH env)")

def _load(cfg):
    path = cfg.getoption("--case") or os.getenv("CASE_PATH")
    if not path or not os.path.isfile(path):
        pytest.skip("provide case via --case or CASE_PATH", allow_module_level=True)
    with open(path) as f:
        return json.load(f)

CFG = _load(pytest)

@pytest.mark.parametrize("cfg", [CFG], ids=[CFG["name"]])
def test_single(cfg):
    batch = [list(range(cfg["prompt_len"]))] * 4          # 4‑sequence batch

    model = DummyBoostModel()
    LazyBatchAllocator(
        batch,
        cfg["workers"],
        model,
        page_size     = cfg["page_size"],
        table_capacity= cfg["table_cap"],
    )
    seen = {w for w, _ in model.calls}
    assert seen == set(cfg["expected_workers"])
