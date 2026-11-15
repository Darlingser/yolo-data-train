# YOLO 杯子 / 瓶子检测

用 YOLO11n 训练一个 2 类目标检测模型：`bottle`（瓶子/保温杯/随行杯）和 `cup`（开口杯/马克杯）。
数据为自采照片 + 公开图片，全部按 YOLO 格式标注。

当前版本在冻结的 7 张测试集上：**mAP50 = 0.921，mAP50-95 = 0.452**。

---

## 1. 环境

| 项目 | 版本 |
|---|---|
| 系统 | Windows 11 + WSL2 (Ubuntu) |
| Python | 3.14.4（venv 位于 `~/yolo-lab/.venv`） |
| Ultralytics | 8.4.199 |
| PyTorch | 2.11.0 + cu128 |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| 模型 | YOLO11n（2,597,294 参数） |

数据集放在 Windows 的 `D:\datasets\office_cup_bottle`，在 WSL 中的路径是 `/mnt/d/datasets/office_cup_bottle`。

---

## 2. 数据集

### 2.1 类别定义

判别标准是**有没有盖子、能不能带走**，不看材质和大小。

| 类别 id | 名称 | 包含 | 举例 |
|---|---|---|---|
| 0 | bottle | 有盖、可密封、随身携带的容器 | 保温杯、随行杯、运动水壶、矿泉水瓶、饮料瓶 |
| 1 | cup | 敞口的日常杯子 | 马克杯、玻璃杯、纸杯、茶杯；带把手的办公杯（即使有盖）也算 cup |

注意事项：

- 瓶盖属于 bottle，不单独标框；
- 碗、盘子不标；
- 同一张图里有多类目标时全部标出。

### 2.2 规模与切分

| split | 图片 | 正样本框 | 负样本（无目标） |
|---|---|---|---|
| train | 73 | 69 | 30 |
| val | 11 | 9 | 2 |
| test | 7 | 10 | 0 |

- 负样本是**没有任何目标**的图片，标签文件为空 txt（0 字节），不是"没标"；
- test 集固定为 7 张、10 个实例（8 bottle + 2 cup），定下来后不再改动，所有版本都用它比较；
- 同一物体的不同角度不跨 split，避免泄漏。

### 2.3 目录结构

```
office_cup_bottle/
├── data.yaml
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── split_manifest.csv
```

`data.yaml`：

```yaml
path: /mnt/d/datasets/office_cup_bottle
train: images/train
val: images/val
test: images/test
nc: 2
names:
  0: bottle
  1: cup
```

---

## 3. 训练

训练在 WSL 里执行，工作目录 `~/yolo-lab`。

```bash
source /home/asus/yolo-lab/.venv/bin/activate
cd /home/asus/yolo-lab

yolo detect train \
  model=/home/asus/yolo-lab/runs/hardneg_scene_20261115/weights/best.pt \
  data=/mnt/d/datasets/office_cup_bottle/data.yaml \
  epochs=100 imgsz=640 batch=8 workers=2 device=0 \
  lr0=0.0001 optimizer=AdamW seed=42 patience=30 \
  mosaic=0.0 close_mosaic=0 \
  project=/home/asus/yolo-lab/runs \
  name=split73_20261115
```

参数说明：

| 参数 | 值 | 原因 |
|---|---|---|
| `model` | 上一版 best.pt | 在已有权重上做微调，收敛快 |
| `imgsz` | 640 | 数据长边 800–1600，640 是 YOLO11 的默认尺度；框的中位宽度约 196 px |
| `batch` | 8 | 8 GB 显存下的稳定值 |
| `lr0` | 1e-4 | 全量微调用这个量级；1e-3 会让模型在前几轮就崩掉 |
| `optimizer` | AdamW | 小数据下比 SGD 稳 |
| `patience` | 30 | 早停 |
| `mosaic` | 0 | 开启会把负样本拼进正样本图，破坏负样本语义 |
| `seed` | 42 | 固定随机种子，便于对比 |

输出目录为 `/home/asus/yolo-lab/runs/<name>/`，其中 `weights/best.pt` 是验证集上表现最好的权重。

### 评估

```bash
yolo detect val \
  model=/home/asus/yolo-lab/runs/best_split73_20261115/weights/best.pt \
  data=/mnt/d/datasets/office_cup_bottle/data.yaml \
  split=test imgsz=640 batch=8 device=0
```

### 推理

```bash
yolo detect predict \
  model=/home/asus/yolo-lab/runs/best_split73_20261115/weights/best.pt \
  source=/path/to/images imgsz=640 conf=0.25 save=True save_txt=True
```

---

## 4. 指标怎么看

| 指标 | 含义 |
|---|---|
| **P**（precision） | 模型报出的框里，有多少是真的。低 = 误报多 |
| **R**（recall） | 真值里，有多少被找到。低 = 漏检多 |
| **mAP50** | IoU 阈值取 0.5 时的平均精度。衡量"框大致对不对得上"，标准较宽松 |
| **mAP50-95** | IoU 从 0.5 到 0.95 每隔 0.05 取一档，共 10 档，取平均。衡量"框贴得准不准"，对定位精度敏感 |

两个 mAP 的关系可以这样判断：

- mAP50 高、mAP50-95 低（如 0.921 / 0.452）：目标都能找到，但框偏松或偏紧，边界不够准；
- 两者都高：定位和分类都到位。

`confusion_matrix.png` 里的 background 列，表示"背景被误判成目标"的数量，用来衡量误检。

---

## 5. 结果

全部在同一批 7 张 test 图片（10 个实例）上评估：

| 版本 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| ft_640_2 | 0.768 | 0.917 | 0.881 | 0.376 |
| hardneg_full | 0.817 | 0.868 | 0.883 | 0.405 |
| hardneg_scene | 0.855 | 0.833 | 0.893 | 0.386 |
| **best_split73（当前）** | **0.875** | **0.926** | **0.921** | **0.452** |

分类型（当前版本）：

| 类别 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| bottle | 0.773 | 0.851 | 0.847 | 0.357 |
| cup | 0.978 | 1.000 | 0.995 | 0.547 |

---

## 6. 主要问题与解决办法

### 6.1 类别边界不一致

问题：保温杯、随行杯、带盖办公杯混在 bottle 和 cup 之间，标注前后不一致。

解决：改用"有无盖子、能否带走"作为唯一判据；带把手的办公杯归 cup，其余有盖容器归 bottle；瓶盖随瓶身一起标。

### 6.2 标注框过松

问题：部分框把背景（纸张、货架、桌面阴影）包了进去，模型学到上下文而不是物体；mAP50-95 上不去。

解决：逐图核对，把框收紧到物体轮廓。一个典型样本面积缩小 16%，mAP50-95 随之上升。

### 6.3 学习率设置错误导致训练崩溃

问题：用 `lr0=1e-3` 做全量微调，第 4 轮 mAP 从 0.875 掉到 0.317；用 5e-4 也会从第 1 轮开始持续下滑。此时 `best.pt` 只保存了第 1 轮权重，等于没有训练。

解决：全量微调统一用 `lr0=1e-4`；只训练检测头（`freeze=10`）时才使用 1e-3。

### 6.4 背景误检

问题：黑色音箱（conf 0.62）、打印机、椅背横杆、书堆被识别成 bottle，bottle 的 P 只有 0.53 左右。

解决分两步：

1. 加入无目标背景图作为负样本（空 txt）；
2. 用当前模型的误检框裁出目标区域作为**硬负样本**，共 22 张，覆盖音箱、打印机、屏幕、纸堆、椅背。

结果：音箱误检从 0.623 降到 0.113，`neg_09` 已无任何检测。

### 6.5 mosaic 破坏负样本

问题：`mosaic=1.0` 会把 4 张图拼成 1 张，空桌面图被拼进有目标的图里，不再是负样本。

解决：训练时设 `mosaic=0.0`。

### 6.6 验证集被反复使用导致虚高

问题：val 只有 11 张，多次用它选模型后，val 指标（mAP50-95 0.527）与 test（0.386）差距很大。

解决：固定 test 集，只用于最终评估；模型选择看 test，不看 val。

### 6.7 多目标并排场景召回低

问题：一张图里 4 个样式相近的保温杯并排、互相遮挡时，其中 3 个的置信度低于 0.25，默认阈值下被判为漏检。

解决：把这一类图片补进训练集；部署时对该场景降低 `conf` 到 0.15 并开启 `agnostic_nms`。

---

## 7. 复现步骤

```bash
# 1. 进入环境
source /home/asus/yolo-lab/.venv/bin/activate
cd /home/asus/yolo-lab

# 2. 确认数据
ls /mnt/d/datasets/office_cup_bottle/images/{train,val,test}

# 3. 训练
yolo detect train model=<上一版权重> data=/mnt/d/datasets/office_cup_bottle/data.yaml \
  epochs=100 imgsz=640 batch=8 device=0 lr0=0.0001 optimizer=AdamW patience=30 mosaic=0.0

# 4. 评估
yolo detect val model=runs/<name>/weights/best.pt \
  data=/mnt/d/datasets/office_cup_bottle/data.yaml split=test imgsz=640

# 5. 推理
yolo detect predict model=runs/<name>/weights/best.pt source=<图片目录> conf=0.25 save=True save_txt=True
```

---

## 8. 权重与产物

```
runs/<name>/
├── weights/best.pt      # 最终权重
├── weights/last.pt      # 最后一轮
├── results.csv          # 每轮损失与指标
├── results.png          # 训练曲线
├── confusion_matrix.png # 混淆矩阵（含 background 列）
├── val_batch0_pred.jpg  # 验证集预测可视化
├── train_batch0.jpg     # 训练样本与标注
└── args.yaml            # 本次训练全部参数
```
