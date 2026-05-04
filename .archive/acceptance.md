# 验收标准 (Acceptance Criteria)

## 总体目标
将PAIR-Former代码库从实验性状态整理为规范、可维护、可复现的研究代码库，同时保证论文中所有实验的可复现性。

## 第一阶段：代码结构规范化

### 1.1 根目录整洁度
**标准**:
- ✅ 根目录Python文件 ≤ 3个（仅保留setup.py等必要文件）
- ✅ 根目录shell脚本 ≤ 2个（仅保留常用快捷脚本）
- ✅ 临时测试文件全部移除或归档
- ✅ 结果JSON文件移到results/目录

**验证方法**:
```bash
ls -1 *.py *.sh *.json 2>/dev/null | wc -l  # 应该 ≤ 5
```

### 1.2 Launch入口点清晰
**标准**:
- ✅ 核心入口点（7个）有清晰文档说明
- ✅ 过时入口点移到src/launch/archive/
- ✅ CLAUDE.md中的命令列表准确
- ✅ 每个入口点有docstring说明用途

**验证方法**:
```bash
ls src/launch/*.py | wc -l  # 核心文件 ≤ 15个
ls src/launch/archive/*.py | wc -l  # 归档文件 ≥ 10个
```

### 1.3 实验配置有序
**标准**:
- ✅ 论文实验配置（~15个）在configs/experiment/
- ✅ 过时配置在configs/experiment/archive/
- ✅ configs/README.md说明配置分类和用途
- ✅ 配置文件命名规范一致

**验证方法**:
```bash
ls configs/experiment/*.yaml | wc -l  # 核心配置 ≤ 25个
test -f configs/README.md && echo "README exists"
```

### 1.4 Checkpoint目录规范
**标准**:
- ✅ checkpoints/README.md说明目录结构
- ✅ 论文相关checkpoint有明确标记
- ✅ 目录命名规范（数据集_模型_版本）

**验证方法**:
```bash
test -f checkpoints/README.md && echo "README exists"
```

### 1.5 Shell脚本组织
**标准**:
- ✅ scripts/按功能分类（train/eval/data/experiments）
- ✅ scripts/README.md说明脚本用途
- ✅ 过时脚本在scripts/archive/

**验证方法**:
```bash
test -f scripts/README.md && echo "README exists"
ls scripts/archive/*.sh 2>/dev/null | wc -l  # 应该 > 0
```

## 第二阶段：文档和实验管理

### 2.1 文档结构清晰
**标准**:
- ✅ docs/目录结构清晰（experiments/archive/guides/）
- ✅ 实验报告在docs/experiments/
- ✅ 历史文档在docs/archive/
- ✅ 根目录markdown文件 ≤ 5个

**验证方法**:
```bash
ls -1 *.md | wc -l  # 应该 ≤ 5
test -d docs/experiments && echo "experiments dir exists"
test -d docs/archive && echo "archive dir exists"
```

### 2.2 主要文档完整性
**标准**:
- ✅ README.md包含：
  - 项目简介
  - 安装指南
  - 快速开始
  - 实验复现步骤
  - 项目结构
  - 引用信息
- ✅ CLAUDE.md准确反映当前命令和结构
- ✅ EXPERIMENTS.md包含所有论文实验的复现步骤
- ✅ 所有文档链接有效

**验证方法**:
```bash
test -f README.md && test -f CLAUDE.md && test -f EXPERIMENTS.md
grep -q "Installation" README.md && echo "Installation section exists"
```

### 2.3 实验结果归档
**标准**:
- ✅ experiments/目录仅包含论文相关实验
- ✅ experiments/archive/包含历史实验
- ✅ experiments/README.md说明实验组织

**验证方法**:
```bash
test -f experiments/README.md && echo "README exists"
test -d experiments/archive && echo "archive exists"
```

## 第三阶段：代码质量提升

### 3.1 代码格式规范
**标准**:
- ✅ 所有Python文件通过black格式检查
- ✅ import语句按isort规范排序
- ✅ 行长度 ≤ 100字符（black默认88，可放宽到100）

**验证方法**:
```bash
black --check src/
isort --check-only src/
```

### 3.2 类型注解覆盖
**标准**:
- ✅ 核心模块（5个）的公共函数有类型注解
- ✅ 函数签名包含参数和返回值类型
- ✅ 复杂类型使用typing模块

**验证方法**:
```bash
grep -r "def.*->.*:" src/models/registry.py | wc -l  # 应该 > 0
grep -r "def.*->.*:" src/em/controller.py | wc -l  # 应该 > 0
```

### 3.3 代码质量指标
**标准**:
- ✅ TODO/FIXME标记已处理或转为issue
- ✅ 无明显重复代码（DRY原则）
- ✅ 关键函数有docstring
- ✅ 复杂逻辑有注释说明

**验证方法**:
```bash
grep -r "TODO\|FIXME" src/ | wc -l  # 应该 = 0 或有对应issue
```

### 3.4 错误处理完善
**标准**:
- ✅ 文件操作有异常处理
- ✅ 配置加载有错误提示
- ✅ 模型加载有checkpoint验证
- ✅ 数据加载有格式检查

**验证方法**:
- 手动审查关键模块的try-except块
- 测试错误场景（如缺失文件）

## 第四阶段：测试和验证

### 4.1 功能完整性
**标准**:
- ✅ Stage 1训练命令可执行（至少运行1个epoch）
- ✅ Stage 2训练命令可执行
- ✅ Stage 3训练命令可执行
- ✅ 评估命令可加载checkpoint并输出结果
- ✅ 所有论文实验配置可加载

**验证方法**:
```bash
# Smoke test - 每个命令运行1个epoch
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline run.num_epochs=1
python -m src.launch.train experiment=CheapCTSNet run.num_epochs=1
python -m src.launch.train_em experiment=miRAW_EM_Pipeline run.num_epochs=1
```

### 4.2 配置正确性
**标准**:
- ✅ 所有核心实验配置通过Hydra验证
- ✅ 配置文件无语法错误
- ✅ 路径引用正确（相对路径）
- ✅ 模型架构名称与registry匹配

**验证方法**:
```bash
python -m src.launch.train experiment=miRAW_EM_Pipeline --cfg job  # 打印配置
```

### 4.3 依赖准确性
**标准**:
- ✅ requirements.txt包含所有必要依赖
- ✅ 依赖版本固定（关键包）
- ✅ 无未使用的依赖
- ✅ 新环境可成功安装

**验证方法**:
```bash
pip install -r requirements.txt  # 在新虚拟环境测试
python -c "import torch; import hydra; import wandb"
```

### 4.4 文档准确性
**标准**:
- ✅ README中的命令可执行
- ✅ CLAUDE.md中的路径正确
- ✅ EXPERIMENTS.md中的步骤完整
- ✅ 所有内部链接有效

**验证方法**:
- 手动执行README中的示例命令
- 检查文档中的文件路径是否存在

## 第五阶段：最终验收

### 5.1 Git仓库状态
**标准**:
- ✅ 所有修改在refactor分支
- ✅ Commit历史清晰（每阶段一个commit）
- ✅ Commit message描述性强
- ✅ 无未追踪的重要文件

**验证方法**:
```bash
git status  # 应该干净
git log --oneline -10  # 查看commit历史
```

### 5.2 向后兼容性
**标准**:
- ✅ 现有checkpoint可正常加载
- ✅ 核心API未破坏性修改
- ✅ 配置文件格式兼容
- ✅ 数据加载逻辑未改变

**验证方法**:
```bash
# 测试加载现有checkpoint
python -m src.launch.eval experiment=miRAW_EM_Pipeline run.checkpoint=checkpoints/BR-MIL/best.pt
```

### 5.3 可复现性保证
**标准**:
- ✅ 论文Table 1实验可复现（配置存在）
- ✅ 论文Table 2实验可复现
- ✅ 论文Figure 2实验可复现
- ✅ 论文Figure 3-4实验可复现
- ✅ 所有实验的checkpoint路径正确

**验证方法**:
- 检查每个论文实验对应的配置文件存在
- 验证checkpoint路径在配置中正确引用

### 5.4 整体质量指标
**标准**:
- ✅ 代码行数减少或持平（移除冗余）
- ✅ 文件数量减少（归档过时文件）
- ✅ 文档覆盖率提升
- ✅ 代码可读性提升（主观评估）

**量化指标**:
```bash
# 前后对比
find src/ -name "*.py" | wc -l  # Python文件数
find . -name "*.md" | wc -l  # 文档数
du -sh .  # 仓库大小
```

## 最终检查清单

### 必须通过的检查
- [ ] 所有核心训练命令可执行
- [ ] 所有核心评估命令可执行
- [ ] 论文实验配置完整
- [ ] README准确完整
- [ ] CLAUDE.md准确完整
- [ ] 代码通过格式检查
- [ ] Git历史清晰
- [ ] 分支已推送到远程

### 可选但推荐的检查
- [ ] 添加了类型注解
- [ ] 添加了单元测试
- [ ] 更新了依赖版本
- [ ] 创建了CONTRIBUTING.md
- [ ] 创建了CHANGELOG.md

## 回归测试

### 关键功能测试
```bash
# 1. 配置加载测试
python -m src.launch.train experiment=miRAW_EM_Pipeline --cfg job

# 2. 模型构建测试
python -c "from src.models.registry import build_model; from omegaconf import OmegaConf; cfg = OmegaConf.create({'arch': 'PairSetTransformerAggregator'}); model = build_model('PairSetTransformerAggregator', cfg, None)"

# 3. Checkpoint加载测试
python -m src.launch.eval experiment=miRAW_EM_Pipeline run.checkpoint=checkpoints/BR-MIL/best.pt

# 4. 数据加载测试
python -c "from src.data.pair_level_dataset import PairLevelDataset; print('Data module imports successfully')"
```

## 成功标准总结

### 必须达成（P0）
1. ✅ 论文所有实验可复现
2. ✅ 核心训练/评估流程可运行
3. ✅ 文档准确完整
4. ✅ 代码结构清晰
5. ✅ Git历史完整

### 高优先级（P1）
1. ✅ 代码格式规范
2. ✅ 根目录整洁
3. ✅ 配置文件有序
4. ✅ 文档结构清晰

### 中优先级（P2）
1. ✅ 类型注解覆盖
2. ✅ 代码质量提升
3. ✅ 错误处理完善

### 低优先级（P3）
1. 单元测试覆盖
2. CI/CD配置
3. 性能优化

只有P0和P1全部达成，才能认为整理任务完成。
