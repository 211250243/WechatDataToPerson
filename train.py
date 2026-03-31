import os
import argparse
# ==========================================
# 🪄 补丁 1：强制 Python 采用 UTF-8 编码，彻底封杀子进程终端输出引发的 GBK 乱码崩溃
# ==========================================
os.environ["PYTHONIOENCODING"] = "utf-8"
# ==========================================
# 🪄 补丁 2：缓解 12G 显存碎片化 (OOM 防御)
# ==========================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# ==========================================
# 🪄 补丁 3：解决 PyTorch 2.10 与 Triton 3.6 的断层
# 凭空捏造被删除的类，完美骗过 PyTorch 编译器，让 5070 Ti 满血运行
# ==========================================
import triton
import triton.backends.compiler
import triton.compiler.compiler
triton.backends.compiler.AttrsDescriptor = object
triton.compiler.compiler.AttrsDescriptor = object
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. 基础参数与路径配置（命令行覆盖；未传则用默认）
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="models/Qwen3-8B", help="本地模型目录")
parser.add_argument("-d", "--dataset", default="data/周珂帆.jsonl", help="训练用 JSONL")
parser.add_argument("-o", "--output", default="checkpoints/outputs", help="训练 checkpoint 目录")
parser.add_argument("-l", "--lora", default="checkpoints/lora", help="LoRA 输出目录")
parser.add_argument("-g", "--gguf", default="checkpoints/gguf", help="GGUF 导出路径前缀")
parser.add_argument("-e", "--epochs", type=float, default=3.0, help="训练轮数，风格不够可试 4～5")
parser.add_argument("--lora-r", type=int, default=32, dest="lora_r", help="LoRA 秩，越大越能拟合口癖（更吃显存）")
parser.add_argument("--lr", type=float, default=2e-4, help="学习率，epoch 增多时可试 1.5e-4")
args = parser.parse_args()
MODEL_PATH = args.model
DATASET_PATH = args.dataset
OUTPUT_DIR = args.output
FINAL_LORA_DIR = args.lora
GGUF_EXPORT_NAME = args.gguf

MAX_SEQ_LENGTH = 2048 # 12G 显存的黄金分割点，足以覆盖绝大多数微信对话上下文

print("🚀 [1/6] 正在加载模型与分词器...")
# ==========================================
# 2. 加载模型 (开启 4-bit 量化)
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # 自动检测（如果你的环境支持 bf16 会自动开启）
    load_in_4bit = True, # 必须为 True，否则 12G 显存无法加载 8B 模型
)

print("🧩 [2/6] 正在注入 LoRA 适配器...")
# ==========================================
# 3. 配置 LoRA 参数
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_r,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = args.lora_r * 2,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Unsloth 的独家显存优化技术，必开
    random_state = 3407,
)

print("📚 [3/6] 正在处理与格式化数据集...")
# ==========================================
# 4. 数据集处理与 Chat Template 应用
# ==========================================
# 直接使用 Qwen3 原生 chat template，不做覆盖，避免训练/推理模板不一致
# 通过 enable_thinking=False 关闭思考链，让微调模型直接生成回复

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        for convo in convos
    ]
    return { "text" : texts }

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

print("🔥 [4/6] 引擎点火，开始训练...")
# ==========================================
# 5. 配置 Trainer 并启动微调
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # 对于短对话，关闭 packing 有时能让模型更好地学习轮次间的停顿
    args = TrainingArguments(
        per_device_train_batch_size = 1,  # 📉 显存护航：降为 1 避免 5070 Ti 溢出
        gradient_accumulation_steps = 8,  # 📈 保持总 batch size 依然为 8
        warmup_steps = 10,                # 预热步数，防止训练初期 Loss 震荡
        num_train_epochs = args.epochs,
        learning_rate = args.lr,
        fp16 = False,                     # 🪄 补丁 4：50 系显卡直接开启高级 BF16
        bf16 = True,
        logging_steps = 5,                # 每 5 步打印一次 Loss
        optim = "adamw_8bit",             # 8-bit 优化器，极大节省显存
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        save_strategy = "epoch",          # 每个 Epoch 保存一次 checkpoints
    ),
)

# 启动训练
trainer_stats = trainer.train()
print(f"✅ 训练完成！耗时: {trainer_stats.metrics['train_runtime']:.2f} 秒")

print("💾 [5/6] 正在保存 LoRA 权重...")
# ==========================================
# 6. 保存 LoRA 权重
# ==========================================
model.save_pretrained(FINAL_LORA_DIR)
tokenizer.save_pretrained(FINAL_LORA_DIR)

print("📦 [6/6] 正在导出 GGUF 格式 (Ollama 专用)...")
# ==========================================
# 7. 一键导出为 GGUF 格式
# ==========================================
# Q4_K_M 是性能与体积平衡最好的量化格式，8B 模型导出后大约 5GB
model.save_pretrained_gguf(GGUF_EXPORT_NAME, tokenizer, quantization_method = "q4_k_m")

print(f"🎉 全部流程结束！你的 GGUF 模型已生成。")