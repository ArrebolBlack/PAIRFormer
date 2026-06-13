"""Checkpoint state-dict key utilities (refactor/2026-06).

Consolidates the *per-key, sequential* prefix-stripping logic that was duplicated
verbatim across several call sites:

  - src/launch/eval_em.py            `_strip_prefix_state_dict`  prefixes ("model.","module.","net.")
  - src/launch/inference_BR-MIL_online.py        (idem)
  - src/launch/inference_Naive_online.py         (idem)
  - src/launch/inference_targetnet_like_online.py(idem)
  - src/trainer/trainer_em.py        `strip_ddp_prefix`          prefixes ("module.",)
  - src/launch/train.py              `load_model_state`          prefixes ("model.","net.")
  - src/utils/dump_cts_embeddings.py inline                      prefixes ("model.","net.")

Each call site passes its ORIGINAL prefix tuple, so behavior is preserved exactly
(verified by tests/test_checkpoint_utils.py against copies of the original inline
implementations).

NOTE: src/launch/eval_pair_selected.py uses a *different* whole-dict common-prefix
stripper (strip a prefix only if ALL keys share it, repeat until stable). That is a
distinct algorithm and is intentionally left untouched to preserve behavior. Unifying
it is a follow-up cleanup item (see MIGRATION.md), not an equivalence-preserving change.
"""
from typing import Any, Dict, Iterable, Tuple

DEFAULT_PREFIXES: Tuple[str, ...] = ("model.", "module.", "net.")


def clean_state_dict_keys(
    state_dict: Dict[str, Any],
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
) -> Dict[str, Any]:
    """Strip leading prefixes from each key, per-key, sequentially.

    For each key, every prefix in ``prefixes`` is checked in order and stripped if
    the (progressively shortened) key starts with it. With a single-element prefix
    tuple this is identical to a "strip first matching prefix" stripper.

    This reproduces — exactly — the original inline strippers when given that site's
    prefix tuple:
      * ("model.","module.","net.") -> eval_em / inference_*_online
      * ("module.",)                -> trainer_em.strip_ddp_prefix
      * ("model.","net.")           -> train.py.load_model_state / dump_cts_embeddings
    """
    prefixes = tuple(prefixes)
    cleaned: Dict[str, Any] = {}
    for k, v in state_dict.items():
        kk = k
        for pref in prefixes:
            if isinstance(kk, str) and kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    return cleaned
