# 代码整理目标 (Code Refactoring Goals)

## 项目背景
PAIR-Former是一个用于miRNA-mRNA功能性靶标预测的Budgeted Relational Multi-Instance Learning (BR-MIL)框架。当前代码库经过多轮实验迭代，积累了大量临时文件、实验配置和文档，需要系统性整理以提高可维护性和可读性。

## 核心原则
1. **保护关键实验**：论文中的所有实验必须可复现
2. **版本控制安全**：所有修改推送到新分支，避免代码丢失
3. **渐进式重构**：每次小步修改，确保可控和可验证
4. **提升代码质量**：规范化、模块化、文档化

## 关键实验清单（基于论文）

### 主要实验（必须保留）
1. **miRAW 10-fold balanced CV** (Table 1)
   - PAIR-Former: F1=0.840±0.022, PR-AUC=0.869±0.031
   - Max pooling baseline: F1=0.798±0.017
   - TargetNet (official): F1=0.779±0.018
   - Mimosa (official): F1=0.788±0.017

2. **deepTargetPro external validation** (Table 2)
   - PAIR-Former (transfer): F1=83.9±3.9%
   - PAIR-Former (full): F1=83.2±3.2%

3. **MTI large-scale benchmark** (Fig 2a)
   - K=64: F1=0.7708
   - K=512: F1=0.7925, PR-AUC=0.8729
   - 420K pairs, 38× larger than miRAW

4. **Budget analysis** (Fig 2)
   - K sweep: {1,8,16,32,64,128,256,512}
   - Truncate@Kmax vs Retrain@K comparison

5. **Pool-size robustness** (Fig 4)
   - Fixed K=64, varying visible pool size n

6. **Runtime analysis** (Fig 3)
   - Online inference: latency, throughput, VRAM
   - Stage-wise breakdown at K=64

7. **Architecture ablations**
   - Instance encoder scaling (MTI CTS-level)
   - Aggregator family comparison (K=64)
   - Set Transformer scaling sweep (K=64)

8. **Cross-domain validation**
   - CAMELYON16 (pathology)
   - Musk2 (molecule)

### 关键配置文件
- `miRAW_EM_Pipeline.yaml` - 主实验配置
- `deepTargetPro_EM_Pipeline.yaml` - 外部验证
- `MTI_train_selected_inst.yaml` - 大规模实验
- `CheapCTSNet.yaml` - Stage 2蒸馏
- `miRAW_TargetNet_Optimized_baseline.yaml` - Stage 1训练

### 关键Checkpoint
- `BR-MIL/` - 主模型
- `CheapCTSNet/` - Stage 2蒸馏模型
- `MTI_TargetNet_Optimized_shard_v2_relabel_top4/` - MTI Stage 1
- `MTI_CheapCTSNet_shard_v1_compact_r4/` - MTI Stage 2

## 整理目标

### 第一阶段：代码结构规范化 ✓
**目标**：清理根目录，整合launch入口点，规范配置文件
- [ ] 清理根目录临时文件（test_*.py, run_*.sh）
- [ ] 归档或删除过时的launch入口点（保留核心7个）
- [ ] 整合实验配置文件（77个→精简到核心配置）
- [ ] 统一checkpoint目录结构（22个→按实验分类）

### 第二阶段：文档和实验管理 ✓
**目标**：整合分散文档，建立清晰的实验记录体系
- [ ] 整合50+个markdown文档到docs/目录
- [ ] 归档历史实验结果到experiments/archive/
- [ ] 更新README.md和CLAUDE.md
- [ ] 建立实验复现指南

### 第三阶段：代码质量提升 ✓
**目标**：提升代码可读性和可维护性
- [ ] 统一代码风格（black格式化）
- [ ] 添加类型注解（关键模块）
- [ ] 移除重复代码
- [ ] 完善错误处理和日志

### 第四阶段：测试和验证 ✓
**目标**：确保重构后代码功能完整
- [ ] 验证核心训练流程可运行
- [ ] 验证关键实验可复现
- [ ] 建立smoke test
- [ ] 更新依赖文件

## 边界和约束

### 允许的操作
- ✅ 删除明显过时的临时文件
- ✅ 重命名和移动文件（保持git历史）
- ✅ 合并重复代码
- ✅ 添加文档和注释
- ✅ 格式化代码

### 禁止的操作
- ❌ 删除论文中引用的实验配置
- ❌ 修改核心算法逻辑
- ❌ 破坏现有checkpoint兼容性
- ❌ 直接推送到main分支

## 验收标准

### 代码结构
- 根目录整洁（≤5个临时文件）
- launch入口点清晰（≤10个核心文件）
- 配置文件有序（按实验分类）
- checkpoint目录规范

### 文档完整性
- README.md准确反映当前状态
- CLAUDE.md包含最新指南
- 实验复现文档完整
- 历史文档已归档

### 代码质量
- 通过black格式检查
- 核心模块有类型注解
- 无明显重复代码
- 错误处理完善

### 功能验证
- 核心训练流程可运行
- 关键实验配置可加载
- Checkpoint可正常加载
- 依赖文件准确

## Git工作流
1. 创建新分支：`refactor/code-cleanup-2026-05`
2. 每个阶段一个commit
3. 阶段完成后push到远程
4. 最终合并前创建PR review

## 时间规划
- 第一阶段：2-3小时
- 第二阶段：1-2小时
- 第三阶段：2-3小时
- 第四阶段：1-2小时
- 总计：6-10小时
