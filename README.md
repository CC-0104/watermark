# Invisible Watermark Embedding Toolkit

这是一个基于 Python/Tkinter 的图像隐形文字水印工具，支持水印嵌入、盲提取、PSNR 评估，以及多种频域和 AI 水印算法。项目默认使用 `invismark_grid`，在 InvisMark 预训练模型基础上做网格化长文本嵌入，并提供针对旋转、裁剪、缩放的鲁棒提取入口。

## 功能特性

- 图形界面：通过 `app.py` 选择图片、输入文本、嵌入水印并预览结果。
- 盲提取：提取时不需要原图，按嵌入算法选择对应模式即可。
- 多算法支持：
  - `trustmark`：短文本/编号场景推荐，抗 JPEG 压缩表现最好；长文本会自动写入本地查询表，并在图片中嵌入短 ID。
  - `invismark_grid`：默认算法，基于 InvisMark 模型按 256x256 网格切片嵌入长文本。
  - `invismark`：原始 InvisMark 模型，适合短消息和高鲁棒场景。
  - `invismark_pro` / `invismark_logpolar`：实验性增强模式，用于旋转鲁棒性探索。
  - `adaptive_dwt`：自研 DWT-QIM 频域算法，画质高、无需模型。
  - `dwt_dct` / `dwt_dct_svd`：兼容第三方传统水印库。
- 质量反馈：嵌入后自动计算 PSNR，便于判断图像质量损失。

## 项目结构

```text
.
├── app.py                 # Tkinter 桌面应用入口
├── watermark_engine.py    # 水印算法统一封装
├── invismark_repo/        # Microsoft InvisMark 子模块
├── models/                # 本地模型目录，权重不进入 GitHub
├── tests/                 # 系统鲁棒性测试脚本
├── requirements.txt       # Python 依赖
└── README.md
```

## 模型文件

模型权重不存放在 GitHub。请从 Hugging Face 下载：

[shelock/watermark-invismark-model](https://huggingface.co/shelock/watermark-invismark-model)

TrustMark Q 模型文件已同步到：

[shelock/watermark-trustmark-model](https://huggingface.co/shelock/watermark-trustmark-model)

下载后放到：

```text
models/paper.ckpt
```

也可以使用 CLI 下载：

```bash
pip install -U huggingface_hub
huggingface-cli download shelock/watermark-invismark-model paper.ckpt --local-dir models --local-dir-use-symlinks False
```

## 安装

建议使用 Python 3.10 或更新版本。

```bash
git clone --recurse-submodules https://github.com/CC-0104/watermark.git
cd watermark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果已经普通 clone，请补拉子模块：

```bash
git submodule update --init --recursive
```

PyTorch 的 CUDA 版本与显卡环境有关。如果需要 GPU 加速，建议按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本安装。

## 运行

```bash
python app.py
```

界面包含两个页签：

1. 嵌入水印：选择原图，输入水印文本，选择算法后点击嵌入。
2. 提取水印：选择带水印图，选择同一算法后点击提取。

默认算法 `invismark_grid` 不需要填写密码和比特长度。传统频域算法可能需要嵌入时返回的比特长度，并且密码需与嵌入时一致。

## 使用建议

- 短文本、编号、追踪 ID 等场景优先使用 `trustmark`，它在当前系统测试中可通过 JPEG95/JPEG85/JPEG70、缩放、模糊、轻噪声、旋转和轻裁剪。
- `trustmark` 的稳定承载长度约为 8 个 ASCII 字符。更长文本会保存到本地 SQLite 注册表 `.wm_trustmark_registry.sqlite`，图片中只嵌入查询 ID；提取时如果本机有该注册表，会自动还原完整文本。如果换到另一台机器，需要同步这个数据库，或接入后端查询服务，否则只能提取到短 ID。
- 优先使用 PNG 输出。`invismark_grid` 可承受轻度 JPEG95 压缩，但 JPEG85/70 这类强压缩仍可能破坏水印。
- `invismark_grid` 适合较长文本，但要求图片尺寸至少能容纳 256x256 网格块。
- `adaptive_dwt` 不依赖模型，适合轻量部署和高画质无损场景；它不适合作为 JPEG/缩放/模糊后的鲁棒水印。
- 对经过旋转、裁剪、缩放处理的图片，可启用鲁棒提取。

## 系统测试

可以运行内置压力测试脚本，自动生成测试图并验证嵌入、提取和常见攻击：

```bash
python tests/watermark_system_test.py --report
```

默认测试 `adaptive_dwt`、`invismark_grid` 和 `trustmark`。直接提取失败会返回非零退出码；JPEG、缩放、模糊、噪声、旋转、裁剪等攻击结果会写入 JSON 和 HTML 报告。HTML 报告包含原图、水印图、放大差异图、攻击样张和提取结果，便于比较不同迭代的鲁棒性。

如果需要查看极限边界值，可以运行边界扫描：

```bash
python tests/watermark_boundary_test.py --methods trustmark
```

该报告会按攻击强度递增扫描 JPEG 质量、中心裁剪、裁剪后缩放、降采样、模糊、噪声、旋转和遮挡，并标记每组攻击的连续稳定边界、首次失败点、零散成功点，以及“没有抛异常但结果不匹配”的静默失败次数。它用于理解当前模型能力范围，不建议把某次边界结果当成所有图片的绝对承诺；生产环境应按业务图片类型定期抽样复测。

## 工业落地架构

推荐生产链路使用 `trustmark + watermark_id + 注册表/数据库`：

```text
嵌入侧：业务 payload -> SQLite/后端注册表 -> 8 位 watermark_id -> TrustMark 嵌入图片
提取侧：图片 -> TrustMark 提取 watermark_id -> 查询注册表 -> 返回完整 payload
```

本地默认使用 SQLite 注册表 `.wm_trustmark_registry.sqlite`，表中保存 `watermark_id`、payload、payload hash、源图 hash、算法、状态、创建/更新时间等字段。可以运行端到端 demo：

```bash
python tests/trustmark_registry_demo.py
```

该 demo 会生成图片、写入 SQLite、嵌入短 ID、模拟 JPEG70 压缩，再提取 ID 并查库还原完整 payload，同时输出 `registry_demo.html`。

## 依赖说明

核心依赖包括：

- `torch` / `torchvision`：InvisMark 模型推理。
- `opencv-python`、`Pillow`、`numpy`、`PyWavelets`：图像处理与频域变换。
- `bchlib`：InvisMark 消息纠错。
- `trustmark`：TrustMark 短文本鲁棒水印。
- `invisible-watermark`、`blind-watermark`：传统水印算法兼容。

## 清理策略

仓库仅保留运行必需代码、依赖声明、子模块声明和说明文档。以下内容不会进入 GitHub：

- `models/` 下的模型权重。
- 本地运行配置 `.wm_config.json`。
- TrustMark 长文本注册表 `.wm_trustmark_registry.sqlite`。
- `__pycache__/`、测试图片和实验性清洗目录。
- 编辑器配置目录 `.vscode/`。

## 致谢

AI 水印模型部分基于 Microsoft 的 [InvisMark](https://github.com/microsoft/InvisMark) 项目。使用或引用相关模型能力时，请同时参考原项目的论文、许可证和引用要求。
