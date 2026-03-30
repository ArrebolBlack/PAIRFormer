#!/usr/bin/env python3
"""
生成 Selector 消融实验的配置文件
EXP-A: miRAWtest K=8, 16
EXP-B: deepTargetPro K=8, 16
"""
import os
from pathlib import Path

# 配置输出目录
BASE_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "experiment" / "selector_ablation"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置
EXPERIMENTS = {
    "miRAW": {
        "base_exp": "miRAW_EM_Pipeline",
        "K_values": [8, 16],
        "num_epochs": 50,
        "warmup_epochs": 25,
    },
    "deepTargetPro": {
        "base_exp": "deepTargetPro_EM_Pipeline",
        "K_values": [8, 16],
        "num_epochs": 60,
        "warmup_epochs": 30,
    },
}

# Selector 变体定义
SELECTORS = {
    "S0": {
        "k1_ratio": 1.0,
        "use_hash_dedup": False,
        "desc": "TopK (score-based only)",
    },
    "S1": {
        "k1_ratio": 0.5,
        "use_hash_dedup": False,
        "desc": "TopK + Position Diversity",
    },
    "S2": {
        "k1_ratio": 0.5,
        "use_hash_dedup": True,
        "desc": "TopK + Position + Embedding Diversity (full STSelector)",
    },
}

def generate_config(dataset, base_exp, K, sel_name, sel_cfg, num_epochs, warmup_epochs):
    """生成单个配置文件"""
    config_name = f"{dataset}_{sel_name}_K{K}.yaml"
    config_path = BASE_DIR / config_name

    content = f"""# @package _global_
# Selector Ablation: {dataset} {sel_name} K={K}
# {sel_cfg['desc']}

defaults:
  - /experiment/{base_exp}

experiment_name: selector_ablation_{dataset}_{sel_name}_K{K}_seed${{seed}}

# Override seed (will be set via command line)
seed: ${{seed}}

# Override K budget
run:
  kmax: {K}
  num_epochs: {num_epochs}

# Override selector config
em:
  selector_module:
    cfg:
      kmax: {K}
      k1_ratio: {sel_cfg['k1_ratio']}
      use_hash_dedup: {str(sel_cfg['use_hash_dedup']).lower()}

  policy:
    warmup_epochs: {warmup_epochs}

# Override paths to isolate caches
paths:
  cache_root: cache/selector_ablation/{dataset}_{sel_name}_K{K}

em_cache_root: cache/selector_ablation/{dataset}_{sel_name}_K{K}

# Override wandb project
logging:
  wandb:
    enabled: true
    project: "rebuttal_selector_ablation"
    group: "{dataset}_K{K}"
    tags: ["selector_ablation", "{dataset}", "{sel_name}", "K{K}"]
"""

    config_path.write_text(content)
    return config_path

def main():
    print("=" * 60)
    print("Generating Selector Ablation Configs")
    print("=" * 60)

    generated_files = []

    for dataset, exp_cfg in EXPERIMENTS.items():
        print(f"\n[{dataset}]")
        for K in exp_cfg["K_values"]:
            for sel_name, sel_cfg in SELECTORS.items():
                config_path = generate_config(
                    dataset=dataset,
                    base_exp=exp_cfg["base_exp"],
                    K=K,
                    sel_name=sel_name,
                    sel_cfg=sel_cfg,
                    num_epochs=exp_cfg["num_epochs"],
                    warmup_epochs=exp_cfg["warmup_epochs"],
                )
                generated_files.append(config_path)
                print(f"  ✓ {config_path.name}")

    print(f"\n{'=' * 60}")
    print(f"Generated {len(generated_files)} config files in:")
    print(f"  {BASE_DIR}")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
