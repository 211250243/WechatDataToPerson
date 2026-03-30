微信聊天记录训练虚拟人物
---

> Qwen3-8B + Unsloth + Ollama 本地5070Ti（12G）部署
整个流程可以分为五个核心阶段：**WeFlow 微信聊天记录导出 -> Python 数据清洗 -> Unsloth 模型微调 Qwen3-8B -> GGUF 格式导出 -> Ollama 本地部署**。

---

## 一、使用说明

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
python data_processing.py --input data/私聊_刘雨欣.csv --output data/刘雨欣.jsonl
# 可选：会话切分间隔（分钟），默认 60
# python data_processing.py -i data/私聊_xxx.csv -o data/xxx.jsonl --split-minutes 60
```

### 4. 训练模型
```bash
python train.py --model models/Qwen3-8B --dataset data/刘雨欣.jsonl
# 可选：checkpoint / LoRA / GGUF 输出位置
# python train.py -m models/Qwen3-8B -d data/刘雨欣.jsonl --output-dir checkpoints/outputs --lora-dir checkpoints/lora --gguf checkpoints/gguf
```

## 二、流程介绍

### 准备阶段：获取数据与模型

使用开源项目 WeFlow / WeClone / WeChatMsg(MemoTrace) / WechatExporter 导出微信聊天记录（csv格式）。在 ModelScope / HuggingFace 下载 Qwen3-8B 模型。

### 第一步：将微信 CSV 转换为训练数据 (ShareGPT 格式)
微信导出的一般是流水账式的 CSV，而大模型微调最常用的是 **ShareGPT 格式**的 JSONL 文件。你需要写一个简单的 Python 脚本，将你（或其他发送者）的话作为 `human`，将你想训练的“虚拟人物”的话作为 `gpt`。

**目标 JSONL 格式示例：**
```json
[
  {
    "conversations": [
      { "from": "human", "value": "今天晚上吃什么？" },
      { "from": "gpt", "value": "老样子，去吃那家牛肉面吧，我都馋了！" }
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

现在，将微调好的模型放入 Ollama 中，并编写一个系统提示词（System Prompt）来固定它的人设。

1. **创建 Modelfile：**
   在你存放 GGUF 文件的目录下，新建一个无后缀的文件，命名为 `Modelfile`，填入以下内容：

```text
# 指向你的 GGUF 文件路径
FROM ./unsloth.Q4_K_M.gguf

# 设置生成参数，让语气更自然
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# 设置系统提示词（赋予虚拟人物灵魂）
SYSTEM """
你是 [虚拟人物的名字]。你的说话风格是 [例如：幽默、喜欢用感叹号、经常怼人但心地善良]。
请严格按照你以往的微信聊天习惯来回答我的问题。不要表现得像一个 AI，不要说“我是人工智能”之类的话。
"""

# 设定对话模板 (匹配 Qwen 格式)
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

2. **构建 Ollama 模型：**
   打开终端，运行：
   ```bash
   ollama create my_wechat_bot -f Modelfile
   ```

3. **开始聊天！**
   ```bash
   ollama run my_wechat_bot
   ```

---
