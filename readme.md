微信聊天记录训练虚拟人物
---

> Qwen3-8B + Unsloth + Ollama 本地5070Ti（12G）部署
整个流程可以分为五个核心阶段：**WeFlow 微信聊天记录导出 -> Python 数据清洗 -> Unsloth 模型微调 Qwen3-8B -> GGUF 格式导出 -> Ollama 本地部署**。

---

## 一、使用说明

```bash
python train.py -d data/刘雨欣.jsonl -l checkpoints/lora_liu -g checkpoints/gguf_liu --lora-r 8 --epochs 1 --lr 1e-4
```

所有指令：

```bash
# 配置环境，下载模型
python -m venv venv
source venv/bin/activate
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B
pip install -r requirements.txt
# 处理数据，训练模型
python data_processing.py --input data/xxx.csv --output data/xxx.jsonl
python train.py --model models/Qwen3-8B --dataset data/xxx.jsonl
python export_gguf.py -l checkpoints/lora -g checkpoints/gguf
# 部署模型，唤醒对话
ollama create xxx -f Modelfile
ollama run xxx
ollama rm xxx
```

### 1. 导出数据并下载模型

使用开源项目 WeFlow / WeClone / WeChatMsg(MemoTrace) / WechatExporter 导出微信聊天记录（csv格式）。

在 ModelScope / HuggingFace 下载 Qwen3-8B 模型：
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B
```

### 2. 安装核心依赖

```bash
pip install -r requirements.txt
```
或手动安装：
```bash
# pip uninstall torch torchvision torchaudio unsloth unsloth_zoo torchao xformers bitsandbytes peft trl transformers accelerate triton triton-windows -y

# 安装底层计算框架，将版本卡在 Unsloth 允许的 2.10.0 上限，同时带上 xformers
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 xformers --index-url https://download.pytorch.org/whl/cu128
# 安装微调生态链及量化组件
pip install unsloth transformers trl datasets peft accelerate sentencepiece protobuf bitsandbytes>=0.43.0
```

### 3. 处理数据

```bash
python data_processing.py --input data/xxx.csv --output data/xxx.jsonl
# 可选：会话切分间隔（分钟），默认 60
# python data_processing.py -i data/xxx.csv -o data/xxx.jsonl --split-minutes 60
```

### 4. 训练、导出模型

```bash
python train.py --model models/Qwen3-8B --dataset data/xxx.jsonl
# 可选：checkpoint / LoRA / GGUF 输出位置
# python train.py -m models/Qwen3-8B -d data/xxx.jsonl -o checkpoints/outputs -l checkpoints/lora -g checkpoints/gguf -e 3 --lora-r 32 --lr 2e-4
```

如果模型训练成功，导出gguf模型失败，则使用以下命令单独导出：

```bash
python export_gguf.py -l checkpoints/lora -g checkpoints/gguf
```

如果单独导出gguf遇到问题，如 llama-quantize.exe 执行失败，需手动编译 llama.cpp

1. 去官网下载 CMake 的 Windows 安装包：https://cmake.org/download/ （选择 cmake-xxx-windows-x86_64.msi）。
2. 下载 Visual Studio Build Tools（微软官方免费的纯编译工具）：https://www.google.com/search?q=https://aka.ms/vs/17/release/vs_BuildTools.exe
  * 勾选“使用 C++ 的桌面开发” (Desktop development with C++)。
3. 重启
4. 编译 llama.cpp
```bash
# 1. 删除旧的损坏 llama.cpp 文件
Remove-Item -Recurse -Force "C:\Users\20960\.unsloth\llama.cpp"
# 2. 重新克隆完整源码（递归下载子模块）
git clone --recursive https://github.com/ggerganov/llama.cpp "C:\Users\20960\.unsloth\llama.cpp"
# 3. 进入目录
cd "C:\Users\20960\.unsloth\llama.cpp"
# 4. CMake 配置项目
cmake -S . -B build -DBUILD_SHARED_LIBS=OFF
# 5. 编译 Release 版本（等待5-10分钟，出现 Build succeeded 即成功）
cmake --build build --config Release
```

### 5. 配置、部署模型

在 `Modelfile` 文件中指定模型路径、配置系统提示词。然后在 Ollama 中注册并打包虚拟人。

```bash
ollama create xxx -f Modelfile
```

### 6. 唤醒对话

```bash
ollama run xxx
```


## 二、流程介绍

### 准备阶段：获取数据与模型

使用开源项目 WeFlow / WeClone / WeChatMsg(MemoTrace) / WechatExporter 导出微信聊天记录（csv格式）。在 ModelScope / HuggingFace 下载 Qwen3-8B 模型。

### 第一步：将微信 CSV 转换为训练数据 (OpenAI 格式)
微信导出的一般是流水账式的 CSV，而大模型微调最常用的是 **OpenAI 格式**的 JSONL 文件。你需要写一个简单的 Python 脚本，将你（或其他发送者）的话作为 `user`，将你想训练的“虚拟人物”的话作为 `assistant`。

**目标 JSONL 格式示例：**
```json
[
  {
    "conversations": [
      {"role": "user", "content": "你好，我是xxx。"},
      {"role": "assistant", "content": "你好，我是xxx。"},
    ]
  }
]
```

> **💡 核心提示：** 这一步最耗时。你需要清洗掉 CSV 中的无意义信息（如撤回提示、表情包代码），并尽量把连续几条同一个人发的消息合并成一条，保证对话的上下文连贯。

### 第二步：使用 Unsloth 进行 QLoRA 微调

由于你的显存是 12GB，**必须开启 4-bit 量化**，并将 `max_seq_length` 控制在 2048 左右，以防 OOM（显存溢出）。在 5070 Ti 上，Unsloth 的训练速度会让你感到惊喜。

### 第三步：将微调后的模型导出为 GGUF 格式

Unsloth 最强大的功能之一就是原生支持直接导出给 Ollama 用的 GGUF 格式。

### 第四步：使用 Ollama 本地部署并注入灵魂

现在，将微调好的模型放入 Ollama 中，并编写一个系统提示词（System Prompt）来固定它的人设。然后构建 Ollama 模型，运行。

---
