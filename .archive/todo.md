# 任务拆解 (Task Breakdown)

## 当前状态概览
- **仓库大小**: 5.1GB
- **Python文件**: 181个
- **Launch入口**: 29个
- **实验配置**: 77个
- **Checkpoint目录**: 22个
- **Shell脚本**: 45个
- **Markdown文档**: 50+个
- **Git状态**: 1个修改文件 (paper/artifacts/figures/fig4_MTI.pdf)

## 第一阶段：代码结构规范化

### 1.1 创建工作分支
- [ ] 创建新分支 `refactor/code-cleanup-2026-05`
- [ ] 确认git状态干净
- [ ] Push初始状态

### 1.2 清理根目录临时文件
**当前根目录临时文件**:
- [ ] `run_cache_build.py` - 检查是否可移到scripts/
- [ ] `run_seed2025_ext150.sh` - 检查是否可移到scripts/
- [ ] `run_seed2026_ext150.sh` - 检查是否可移到scripts/
- [ ] `test_config_and_checkpoint.py` - 移到tests/或删除
- [ ] `update_model_name.sh` - 移到scripts/或删除
- [ ] `test_checkpoint_update.yaml` - 移到tests/或删除
- [ ] `eval_pair_selected_*.json` - 移到results/或删除
- [ ] `exp9_results_summary.json` - 移到results/或删除

### 1.3 整合launch入口点
**核心入口点（保留）**:
- `train.py` - Stage 1-2训练
- `train_em.py` - Stage 3训练
- `train_pair_selected_inst.py` - Pair-level训练
- `eval.py` - Stage 1-2评估
- `eval_em.py` - Stage 3评估
- `eval_pair_selected.py` - Pair-level评估
- `bench_compute_vs_k.py` - Runtime分析

**需要审查的入口点（29个）**:
- [ ] 审查每个launch/*.py文件用途
- [ ] 标记论文实验相关的文件
- [ ] 归档或删除过时文件到archive/
- [ ] 更新CLAUDE.md中的命令列表

### 1.4 整合实验配置文件
**核心配置（必须保留）**:
- `miRAW_EM_Pipeline.yaml`
- `deepTargetPro_EM_Pipeline.yaml`
- `MTI_train_selected_inst.yaml`
- `CheapCTSNet.yaml`
- `miRAW_TargetNet_Optimized_baseline.yaml`

**配置整理任务**:
- [ ] 列出所有77个实验配置
- [ ] 按数据集分类（miRAW/deepTargetPro/MTI/其他）
- [ ] 标记论文实验相关配置
- [ ] 归档过时配置到configs/archive/
- [ ] 创建configs/README.md说明配置用途

### 1.5 规范checkpoint目录
**当前22个checkpoint目录**:
- [ ] 审查每个checkpoint用途
- [ ] 标记论文实验相关checkpoint
- [ ] 创建checkpoints/README.md说明结构
- [ ] 考虑是否需要Git LFS管理

### 1.6 整合shell脚本
**当前45个shell脚本**:
- [ ] 审查scripts/目录结构
- [ ] 按功能分类（训练/评估/数据处理/实验）
- [ ] 归档过时脚本到scripts/archive/
- [ ] 创建scripts/README.md

## 第二阶段：文档和实验管理

### 2.1 整合markdown文档
**当前50+个文档分布**:
- 根目录: CLAUDE.md, README.md, EXP4_Report.md, EXPERIMENT_REPORT.md等
- docs/: 7个文档
- experiments/: 多个实验报告
- scripts/rebuttal/: 多个实验文档

**整理任务**:
- [ ] 创建docs/experiments/目录
- [ ] 移动实验报告到docs/experiments/
- [ ] 创建docs/archive/归档历史文档
- [ ] 整合重复内容
- [ ] 更新文档索引

### 2.2 更新主要文档
- [ ] 更新README.md（反映当前状态）
- [ ] 更新CLAUDE.md（移除过时命令）
- [ ] 创建EXPERIMENTS.md（实验复现指南）
- [ ] 创建CONTRIBUTING.md（开发指南）

### 2.3 归档实验结果
- [ ] 创建experiments/archive/目录
- [ ] 移动历史实验到archive/
- [ ] 保留论文相关实验在experiments/
- [ ] 创建experiments/README.md

## 第三阶段：代码质量提升

### 3.1 代码格式化
- [ ] 安装black和isort
- [ ] 格式化src/目录所有Python文件
- [ ] 格式化scripts/目录核心脚本
- [ ] 提交格式化commit

### 3.2 添加类型注解
**优先级模块**:
- [ ] src/models/registry.py
- [ ] src/em/controller.py
- [ ] src/selectors/st_selector.py
- [ ] src/trainer/trainer_em.py
- [ ] src/data/em_cache.py

### 3.3 代码审查和优化
- [ ] 检查TODO/FIXME标记（5个文件）
- [ ] 识别重复代码
- [ ] 优化import语句
- [ ] 添加docstring（关键函数）

### 3.4 错误处理
- [ ] 审查关键模块的异常处理
- [ ] 添加有意义的错误消息
- [ ] 统一日志格式

## 第四阶段：测试和验证

### 4.1 功能验证
- [ ] 测试Stage 1训练命令
- [ ] 测试Stage 2训练命令
- [ ] 测试Stage 3训练命令
- [ ] 测试评估命令
- [ ] 验证checkpoint加载

### 4.2 配置验证
- [ ] 验证核心实验配置可加载
- [ ] 检查配置文件语法
- [ ] 验证路径引用正确

### 4.3 依赖管理
- [ ] 审查requirements.txt
- [ ] 移除未使用的依赖
- [ ] 固定关键依赖版本
- [ ] 测试环境安装

### 4.4 文档验证
- [ ] 验证README命令可执行
- [ ] 检查文档链接有效性
- [ ] 确保实验复现步骤完整

## 第五阶段：最终整理和提交

### 5.1 最终检查
- [ ] 运行smoke test
- [ ] 检查git状态
- [ ] 审查所有修改
- [ ] 更新CHANGELOG

### 5.2 提交和推送
- [ ] 提交所有修改
- [ ] Push到远程分支
- [ ] 创建PR（如需要）
- [ ] 更新项目文档

## 风险和注意事项

### 高风险操作
- 删除checkpoint文件（需要确认不在论文中）
- 删除实验配置（需要确认不在论文中）
- 修改核心算法代码

### 回滚计划
- 每个阶段独立commit
- 保持分支可随时回滚
- 重要文件先备份

## 依赖关系
- 阶段1.2-1.6可并行
- 阶段2依赖阶段1完成
- 阶段3可与阶段2并行
- 阶段4依赖阶段1-3完成
