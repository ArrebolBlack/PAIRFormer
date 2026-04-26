#!/usr/bin/env python3
"""Update model names to glm-5.1.2 in config files."""
import sys
import yaml
from pathlib import Path

CONFIG_FILES = [
    "configs/experiment/MTI_TargetNet_Optimized.yaml",
    "configs/experiment/deepTargetPro_TargetNet_Optimized.yaml",
]

OLD_NAME = "targetnet_optimized"
NEW_NAME = "glm-5.1.2"

def update_config_file(config_path: str):
    """Update model name in a single config file."""
    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Replace all occurrences of OLD_NAME with NEW_NAME
        updated = content.replace(OLD_NAME, NEW_NAME)

        with open(config_path, 'w') as f:
            f.write(updated)

        print(f"✓ Updated: {config_path}")
        return True
    except Exception as e:
        print(f"✗ Failed: {config_path} - {e}")
        return False

def main():
    print("=== 开始修改配置文件 ===")
    print(f"Old name: {OLD_NAME}")
    print(f"New name: {NEW_NAME}")

    updated_count = 0
    for config_path in CONFIG_FILES:
        if Path(config_path).exists():
            if update_config_file(config_path):
                updated_count += 1
        else:
            print(f"⚠ Not found: {config_path}")

    print(f"\n=== 修改完成 ===")
    print(f"成功更新了 {updated_count} 个配置文件")

if __name__ == "__main__":
    main()
