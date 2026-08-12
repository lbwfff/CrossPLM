# CrossPLM

**Mechanistic Interpretability for Cross-Task Protein Language Models**

蛋白语言模型（PLM）在多个生物学任务上表现出色，但其内部机制尚未被充分理解。本项目旨在构建跨任务 PLM 的可解释性分析框架，包含快速 fine-tuning 工具链和可解释性研究模块。

---

## 项目结构

```
CrossPLM/
├── Dataset/
│   └── relaxdb_data.csv          # RelaxDB 数据集（蛋白主链动力学）
├── Training/                     # 快速 fine-tuning 具体任务的 PLM 模型
│   ├── launch.py                 # 训练启动入口（两阶段：generate / train）
│   ├── requirements.txt          # 依赖
│   ├── config/
│   │   ├── __init__.py
│   │   └── training_config.py    # TrainingConfig 数据类 + YAML 模板生成
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # TokenClassificationDataset + 自动 class weight
│   ├── models/
│   │   ├── __init__.py
│   │   └── plm_model.py          # HuggingFace 模型封装 (MLM / TokenClassification)
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── trainer.py            # 训练循环 + F1 + Top-k checkpoints + 曲线图
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py         # 任务目录创建、配置文件读写
│   ├── scripts/
│   │   └── preprocess_relaxdb.py # RelaxDB → 训练用 CSV 转换脚本
│   ├── examples/
│   │   ├── sample.csv            # 示例数据
│   │   ├── sample.fasta
│   │   ├── sample_config.yaml
│   │   └── relaxdb_processed.csv # 预处理后的 RelaxDB 数据
│   └── tasks/                    # 运行后自动生成的任务目录（含 config / checkpoints / 曲线图）
└── Crossing/                     # 跨任务 PLM 可解释性研究（待开发）
```

---

## Training — 功能概览

### 使用流程

两阶段设计：

```bash
cd Training

# 阶段 1：生成任务模板
python launch.py --task_name my_experiment
# → 自动创建 tasks/my_experiment_<timestamp>/config.yaml

# 编辑 config.yaml 填写数据路径、超参数等

# 阶段 2：启动训练
python launch.py --config tasks/my_experiment_<timestamp>/config.yaml
```

### 核心功能

| 功能 | 说明 |
|---|---|
| **两阶段启动** | 先模板 → 编辑 → 再训练，配置与代码分离 |
| **CSV 数据** | `sequence` + `label` 两列，自动按 `train_ratio` 划分 |
| **标签忽略** | 支持 `_` 作为 ignore 标记（对应位置不参与 loss） |
| **自动 class weight** | 支持 `none` / `inverse` / `log` 三种权重策略 |
| **评价指标** | Loss + Accuracy + Macro F1 |
| **Top-k checkpoints** | 按 F1 保留最优的 3 个 checkpoint |
| **训练曲线图** | 训练结束后自动生成 epoch–F1 曲线 (`training_curve.png`) |

### 配置参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `backbone_model_id` | HuggingFace 模型 ID | `facebook/esm2_t6_8M_UR50D` |
| `csv_data_path` | CSV 数据路径（必填） | — |
| `task_type` | `token_classification` / `mlm` | `token_classification` |
| `class_weight_method` | 类别权重策略 | `inverse` |
| `train_ratio` | 训练集比例 | 0.9 |
| `num_train_epochs` | 训练轮数 | 20 |
| `learning_rate` | 学习率 | 5e-5 |

### 当前基线（RelaxDB, ESM2-8M）

| 指标 | 值 |
|---|---|
| Accuracy | ~0.80 |
| Macro F1 | ~0.62 |
| 类别分布 | class 0: 12,724 / class 1: 1,535（8:1） |

### 依赖

```bash
pip install -r Training/requirements.txt
```

---

## Crossing — 当前进展

**尚未开始。** 规划方向包括：

- 跨任务 PLM 神经元 / 注意力头的重要性分析
- 任务共享 vs 任务特有表示的分离与可视化
- 因果干预（activation patching / interchange interventions）
- 基于 probing 的表示可迁移性分析

---

## 路线图

- [x] Training 两阶段启动框架
- [x] CSV 数据加载与自动划分
- [x] Token classification 训练（MLM 可选）
- [x] 自动 class weight 计算（inverse / log）
- [x] Macro F1 评估与 Top-3 checkpoint
- [x] 训练曲线可视化
- [x] RelaxDB 数据预处理脚本
- [ ] Crossing 可解释性分析框架
- [ ] 更多 backbone 支持（ProtBERT, Ankh 等）
- [ ] 跨任务神经元归因工具
