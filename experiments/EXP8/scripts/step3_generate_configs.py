#!/usr/bin/env python3
"""
EXP8 Step 3: Generate Hydra config files for MTI experiments
Author: Auto-generated for ICML 2026 Rebuttal
Date: 2026-03-29
"""

import yaml
from pathlib import Path
import argparse
import shutil


def generate_data_config(output_dir, data_file='MTI_pair_random_split.txt'):
    """Generate configs/data/miRNA_MTI.yaml"""
    config = {
        'name': 'mirna_MTI',
        'path': {
            'train': f'data/MTI/{data_file}',
            'val': f'data/MTI/{data_file}',
            'test': f'data/MTI/{data_file}'
        },
        'with_esa': True,
        'split_column': 5,  # Assuming 'split' is the 6th column (0-indexed: 5)
        'split_map': {
            'train': 'train',
            'val': 'val'
        }
    }

    output_path = output_dir / 'data' / 'miRNA_MTI.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {output_path}")
    return output_path


def copy_and_modify_experiment_config(
    template_path,
    output_path,
    data_name='miRNA_MTI',
    experiment_name=None,
    instance_ckpt_path=None,
    cheap_ckpt_path=None
):
    """Copy experiment config and modify data reference"""

    # Load template
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify data reference
    if 'data' in config:
        config['data']['name'] = data_name

    # Modify experiment name
    if experiment_name and 'experiment' in config:
        config['experiment']['name'] = experiment_name
    if experiment_name:
        config['experiment_name'] = experiment_name

    # Modify checkpoint paths
    if instance_ckpt_path:
        config['instance_ckpt_path'] = instance_ckpt_path
    if cheap_ckpt_path:
        config['cheap_ckpt_path'] = cheap_ckpt_path

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {output_path}")
    return output_path


def generate_all_configs(configs_root, templates_root):
    """Generate all config files for MTI experiments"""

    print("Generating config files for MTI experiments...\n")

    # 1. Data config
    generate_data_config(configs_root)

    # 2. Stage 1: TargetNet_Optimized
    template = templates_root / 'experiment' / 'miRAW_TargetNet_Optimized.yaml'
    output = configs_root / 'experiment' / 'MTI_TargetNet_Optimized.yaml'
    copy_and_modify_experiment_config(
        template,
        output,
        data_name='miRNA_MTI',
        experiment_name='MTI_TargetNet_Optimized'
    )

    # 3. Stage 2: CheapCTSNet
    template = templates_root / 'experiment' / 'CheapCTSNet.yaml'
    output = configs_root / 'experiment' / 'MTI_CheapCTSNet.yaml'
    copy_and_modify_experiment_config(
        template,
        output,
        data_name='miRNA_MTI',
        experiment_name='MTI_CheapCTSNet',
        instance_ckpt_path='checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt'
    )

    # 4. Stage 3: EM Pipeline
    template = templates_root / 'experiment' / 'deepTargetPro_EM_Pipeline.yaml'
    output = configs_root / 'experiment' / 'MTI_EM_Pipeline.yaml'
    copy_and_modify_experiment_config(
        template,
        output,
        data_name='miRNA_MTI',
        experiment_name='MTI_EM_Pipeline',
        instance_ckpt_path='checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt',
        cheap_ckpt_path='checkpoints/MTI_CheapCTSNet/checkpoints/last.pt'
    )

    print("\n" + "="*60)
    print("All config files generated successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate config files for MTI experiments')
    parser.add_argument('--configs_root', type=str,
                        default='configs',
                        help='Root directory for configs')
    parser.add_argument('--templates_root', type=str,
                        default='configs',
                        help='Root directory for template configs')

    args = parser.parse_args()

    configs_root = Path(args.configs_root)
    templates_root = Path(args.templates_root)

    generate_all_configs(configs_root, templates_root)


if __name__ == '__main__':
    main()
