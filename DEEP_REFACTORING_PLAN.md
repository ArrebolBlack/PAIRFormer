# PAIR-Former深度重构方案

## 问题诊断

### 当前问题
1. **目录结构不专业** - 不像成熟论文代码库
2. **临时文件清理不彻底** - 只是移动到temp/，没有删除
3. **核心功能不突出** - 29个launch入口点，使用复杂
4. **代码质量粗糙** - 只做了格式化，没有深入优化

### 对标目标
参考顶级论文代码库：
- **CLIP** (OpenAI): 清晰的模块划分，简单的API
- **Stable Diffusion**: 专业的目录结构，完善的文档
- **LLaMA**: 核心功能突出，易于使用

---

## 深度重构计划

### Phase 1: 目录结构重构 ⚡

#### 目标结构（参考顶级项目）
```
PAIRFormer/
├── README.md                    # 简洁的主文档
├── LICENSE                      # 开源协议
├── setup.py                     # 安装脚本
├── requirements.txt             # 依赖
│
├── pairformer/                  # 核心包（小写，可pip安装）
│   ├── __init__.py
│   ├── models/                  # 模型定义
│   ├── data/                    # 数据处理
│   ├── training/                # 训练逻辑
│   ├── evaluation/              # 评估逻辑
│   └── utils/                   # 工具函数
│
├── scripts/                     # 核心脚本（精简到5-7个）
│   ├── train_stage1.py          # Stage 1训练
│   ├── train_stage2.py          # Stage 2训练
│   ├── train_stage3.py          # Stage 3训练
│   ├── evaluate.py              # 评估
│   └── inference.py             # 推理
│
├── configs/                     # 配置文件
│   ├── default.yaml             # 默认配置
│   └── experiments/             # 实验配置
│
├── examples/                    # 使用示例
│   ├── quickstart.py            # 快速开始
│   ├── train_miRAW.py           # miRAW训练示例
│   └── train_MTI.py             # MTI训练示例
│
├── tests/                       # 测试
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
│
├── docs/                        # 文档
│   ├── installation.md
│   ├── training.md
│   └── api.md
│
├── checkpoints/                 # 预训练模型
└── data/                        # 数据（.gitignore）
```

#### 关键改进
1. **核心包重命名**: src/ → pairformer/ (小写，可pip安装)
2. **精简scripts**: 29个 → 5-7个核心脚本
3. **添加examples/**: 展示核心用法
4. **添加tests/**: 测试框架
5. **删除temp/**: 彻底清理临时文件
6. **精简根目录**: 只保留README, LICENSE, setup.py等

### Phase 2: 代码深度优化 ⚡

#### 2.1 类型注解
为核心模块添加完整类型注解：
- models/registry.py
- training/trainer.py
- data/dataset.py
- evaluation/evaluator.py

#### 2.2 错误处理
- 添加自定义异常类
- 完善错误消息
- 添加输入验证

#### 2.3 性能优化
- 优化数据加载（DataLoader workers）
- 优化缓存机制
- 内存优化（del不用的变量）

#### 2.4 DDP功能验证和优化
- 验证DDP训练可用
- 优化同步机制
- 添加DDP测试

### Phase 3: 核心功能突出 ⚡

#### 3.1 简化入口点
**当前**: 29个launch文件  
**目标**: 5-7个核心脚本

核心脚本：
1. `scripts/train_stage1.py` - Stage 1训练
2. `scripts/train_stage2.py` - Stage 2训练
3. `scripts/train_stage3.py` - Stage 3训练
4. `scripts/evaluate.py` - 评估
5. `scripts/inference.py` - 推理
6. `scripts/build_cache.py` - 缓存构建（可选）
7. `scripts/benchmark.py` - 性能测试（可选）

#### 3.2 创建高级API
```python
# examples/quickstart.py
from pairformer import PAIRFormer, Trainer

# 简单的训练接口
model = PAIRFormer.from_pretrained("checkpoints/BR-MIL")
trainer = Trainer(model, config="configs/default.yaml")
trainer.train()
```

#### 3.3 清晰的三阶段流程
```bash
# 一键训练三阶段
bash scripts/train_all_stages.sh

# 或分步训练
python scripts/train_stage1.py --config configs/miRAW.yaml
python scripts/train_stage2.py --config configs/miRAW.yaml
python scripts/train_stage3.py --config configs/miRAW.yaml
```

### Phase 4: 文档和测试 ⚡

#### 4.1 完善文档
- README.md: 简洁清晰，突出核心功能
- docs/installation.md: 详细安装指南
- docs/training.md: 训练指南
- docs/api.md: API文档

#### 4.2 建立测试框架
- 单元测试（pytest）
- 集成测试
- DDP测试
- CI/CD配置

---

## 执行计划

### Round 1: 彻底清理临时文件
- 删除temp/目录
- 删除明确无用的launch文件
- 删除实验性代码

### Round 2: 重构目录结构
- 重命名src/ → pairformer/
- 创建examples/
- 创建tests/
- 精简scripts/

### Round 3: 精简launch入口点
- 合并相似功能
- 创建5-7个核心脚本
- 移除冗余代码

### Round 4: 代码深度优化
- 添加类型注解
- 完善错误处理
- 性能优化
- DDP验证

### Round 5: 创建高级API和示例
- 设计简洁的API
- 创建examples/
- 编写使用文档

### Round 6: 测试和验证
- 建立测试框架
- 验证核心功能
- DDP测试

---

## 验收标准

### 目录结构
- ✅ 像顶级论文代码库
- ✅ 没有temp/目录
- ✅ 根目录整洁（≤5个文件）
- ✅ 有examples/和tests/

### 核心功能
- ✅ 入口点≤7个
- ✅ 三阶段流程清晰
- ✅ 有高级API
- ✅ 有使用示例

### 代码质量
- ✅ 核心模块有类型注解
- ✅ 完善的错误处理
- ✅ 性能优化
- ✅ DDP功能验证可用

### 文档和测试
- ✅ README简洁清晰
- ✅ 有完整的文档
- ✅ 有测试框架
- ✅ 有CI/CD配置

---

## 时间估计
- Round 1-2: 2-3小时
- Round 3: 2-3小时
- Round 4: 3-4小时
- Round 5: 2-3小时
- Round 6: 2-3小时
- **总计**: 11-16小时

---

## 风险评估
- **高风险**: 重命名src/可能破坏import
- **中风险**: 删除launch文件可能影响某些实验
- **建议**: 在新分支上操作，充分测试后再合并
