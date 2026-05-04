import sys

sys.argv = [
    'run_cache_build.py',
    'experiment=MTI_EM_Scalable_selected_raw_parallel',
    'scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl',
    'cheap_ckpt_path=checkpoints/CheapCTSNet/checkpoints/best.pt',
    'run.split=train',
    'run.kmax=64',
    'scalable.num_pairs_hint=480000',
]

from src.launch import build_selected_pair_cache_parallel
build_selected_pair_cache_parallel.main()
