"""Golden equivalence tests for src.utils.checkpoint.clean_state_dict_keys.

These assert the consolidated utility reproduces, byte-for-byte, the behavior of
the ORIGINAL inline prefix-strippers it replaces. The oracles below are verbatim
copies of the pre-refactor implementations. No torch / GPU / data needed —
state-dict values are plain sentinels, only keys matter.

Run directly:   python tests/test_checkpoint_utils.py
Or via pytest:  pytest tests/test_checkpoint_utils.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.checkpoint import clean_state_dict_keys  # noqa: E402


# ---- oracles: verbatim copies of the original inline implementations ----

def oracle_eval_em(sd):  # eval_em.py / inference_*_online.py
    cleaned = {}
    for k, v in sd.items():
        kk = k
        for pref in ("model.", "module.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    return cleaned


def oracle_trainer_em(sd):  # trainer_em.py strip_ddp_prefix, applied per key
    def strip_ddp_prefix(k):
        for prefix in ("module.",):
            if k.startswith(prefix):
                return k[len(prefix):]
        return k
    return {strip_ddp_prefix(k): v for k, v in sd.items()}


def oracle_train_py(sd):  # train.py load_model_state / dump_cts_embeddings
    cleaned = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("model."):
            kk = kk[len("model."):]
        if kk.startswith("net."):
            kk = kk[len("net."):]
        cleaned[kk] = v
    return cleaned


# A comprehensive set of crafted keys, incl. stacked prefixes and edge cases.
_KEYSETS = [
    ["encoder.weight", "head.bias"],
    ["module.encoder.weight", "module.head.bias"],
    ["model.encoder.weight", "model.head.bias"],
    ["net.encoder.weight", "net.head.bias"],
    ["module.encoder.weight", "head.bias", "model.x", "net.y"],
    ["model.module.x", "module.model.y", "model.net.z", "net.model.w"],
    ["modulething.x", "modelhead.y", "network.z"],  # near-prefixes that must NOT strip
    ["module.module.x"],  # stacked same prefix
    [],
]


def _sd(keys):
    return {k: i for i, k in enumerate(keys)}


def test_default_prefixes_match_eval_em():
    for keys in _KEYSETS:
        sd = _sd(keys)
        assert clean_state_dict_keys(sd) == oracle_eval_em(sd), keys


def test_module_only_matches_trainer_em():
    for keys in _KEYSETS:
        sd = _sd(keys)
        assert clean_state_dict_keys(sd, ("module.",)) == oracle_trainer_em(sd), keys


def test_model_net_matches_train_py():
    for keys in _KEYSETS:
        sd = _sd(keys)
        assert clean_state_dict_keys(sd, ("model.", "net.")) == oracle_train_py(sd), keys


def test_values_pass_through_unchanged():
    sd = {"module.a": 1, "b": 2, "model.c": 3}
    out = clean_state_dict_keys(sd)
    assert out == {"a": 1, "b": 2, "c": 3}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"OK — {len(tests)} checkpoint-util golden tests passed")
