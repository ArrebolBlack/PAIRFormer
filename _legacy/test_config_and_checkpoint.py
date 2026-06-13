#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, '/vepfs-mlp2/mlp-public/haoce/yjq/PAIRFormer')

# 检查配置文件
try:
    with open('configs/experiment/MTI_build_selected_inst.yaml', 'r') as f:
        print("=== 配置文件内容 ===")
        for i, line in enumerate(f, 1):
            print(f"{i: {line.rstrip()}")
except Exception as e:
    print(f"✗ 无法读取配置文件: {e}")

# 检查 checkpoint 文件
target_ckpt = 'checkpoints/TargetNet_officical_pretrained_model.pt'
cheap_ckpt = 'checkpoints/MTI_A_CheapCTSNet/checkpoints/best.pt'

print("\n=== 检查 checkpoint 文件 ===")
print(f"Target checkpoint: {target_ckpt}")
print(f"  存在: {os.path.exists(target_ckpt)} ({'YES' if os.path.exists(target_ckpt) else 'NO'})")

print(f"  大小: {os.path.getsize(target_ckpt) / (1024*1024):.1f} MB" if os.path.exists(target_ckpt) else "N/A")

print(f"\nCheap checkpoint: {cheap_ckpt}")
print(f"  存在: {os.path.exists(cheap_ckpt)} ({'YES' if os.path.exists(cheap_ckpt) else 'NO'})")

print(f"  大小: {os.path.getsize(cheap_ckpt) / (1024*1024):.1f} MB" if os.path.exists(cheap_ckpt) else "N/A")

print("\n=== 建议的配置 ===")
print("如果 Target checkpoint 不可用或大小异常，建议：")
print("1. 使用 CheapCTSNet checkpoint（可靠）")
print("2. 或从源服务器拉取正确的 TargetNet checkpoint")
print("3. 或使用 TargetNet_officical_pretrained_model.pt（如果完整）")

print("\n=== DDP 测试 ===")
print("运行以下命令来测试 DDP：")
print("CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_ddp_train_pair_selected.sh 4")
