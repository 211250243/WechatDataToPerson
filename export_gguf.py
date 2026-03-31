import os
import argparse
# 魔法补丁：强制 Python 采用 UTF-8 编码处理底层日志，彻底封杀 GBK 乱码问题
os.environ["PYTHONIOENCODING"] = "utf-8"

from unsloth import FastLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lora", default="checkpoints/lora", help="LoRA 模型目录")
parser.add_argument("-g", "--gguf", default="checkpoints/gguf", help="GGUF 导出路径前缀")
args = parser.parse_args()
LORA_PATH = args.lora
GGUF_PATH = args.gguf

print("🚀 [1/3] 正在加载你训练好的 LoRA 权重...")
# 这里直接指向你已经保存好的 lora 目录
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_PATH, 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("📦 [2/3] 正在调用 Unsloth 原生编译链导出 GGUF...")
# 这一步会自动拉取 llama.cpp 并调用你刚装好的 CMake 进行编译和量化
model.save_pretrained_gguf(
    GGUF_PATH, 
    tokenizer, 
    quantization_method = "q4_k_m"
)

print("🎉 [3/3] 完美通关！你的 GGUF 模型已成功生成！")