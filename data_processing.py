import argparse
import csv
import json
import re
from datetime import datetime, timedelta

def build_training_dataset(csv_path, output_jsonl_path, split_minutes=60):
    # --- 1. 配置规则 ---
    SYSTEM_KEYWORDS = [
        "朋友验证", "撤回了一条消息", "撤回", "系统消息",
        "对方正在输入", "已撤回", "你添加了", "添加了微信",
        "[表情包]", "[图片]", "[语音]", "[视频]", "[文件]", "[链接]"
    ]
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    # 🌟 核心修改 1：直接采用 OpenAI 格式的角色名
    # is_sender=1 是用户，is_sender=0 是虚拟人物（或者反之，取决于你的训练目标）
    ROLE_SENDER = "user"       
    ROLE_RECEIVER = "assistant" 

    raw_messages = []
    
    # --- 2. 数据读取与初步清洗 ---
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type_name'] != 'text':
                continue
                
            msg = row['msg'].strip()
            # 过滤空消息和系统消息
            if not msg or any(kw in msg for kw in SYSTEM_KEYWORDS):
                continue
                
            # 清理URL和多余空格
            msg = url_pattern.sub('', msg)
            msg = re.sub(r'\s+', ' ', msg).strip()
            if not msg: continue

            try:
                msg_time = datetime.fromisoformat(row['CreateTime'].replace('Z', '+00:00'))
            except ValueError:
                continue

            raw_messages.append({
                "time": msg_time,
                "role": ROLE_SENDER if int(row['is_sender']) == 1 else ROLE_RECEIVER,
                "content": msg
            })

    # 按时间严格排序
    raw_messages.sort(key=lambda x: x['time'])

    # --- 3. 核心逻辑：按时间窗口切片 & 合并同角色发言 ---
    openai_format_data = []
    current_convo = []
    
    for i, msg in enumerate(raw_messages):
        # 1. 判断是否需要开启新一轮对话（时间间隔过长）
        if i > 0:
            time_diff = msg['time'] - raw_messages[i-1]['time']
            if time_diff > timedelta(minutes=split_minutes):
                # 保存上一轮对话（只保存有来有回的，长度>=2）
                if len(current_convo) >= 2:
                    openai_format_data.append({"conversations": current_convo})
                current_convo = [] # 清空，开启新对话

        # 2. 🌟 核心修改 2：键名全部改为 role 和 content
        if current_convo and current_convo[-1]['role'] == msg['role']:
            # 合并同角色连续发言，用换行符拼接
            current_convo[-1]['content'] += '\n' + msg['content']
        else:
            current_convo.append({
                "role": msg['role'],
                "content": msg['content']
            })

    # 收尾：保存最后一轮对话
    if len(current_convo) >= 2:
        openai_format_data.append({"conversations": current_convo})

    # --- 4. 导出为 JSONL 格式 ---
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in openai_format_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ 清洗与切片完成 (已全面转换为原生 OpenAI 格式)！")
    print(f"提取出有效的对话组数量 (Contexts): {len(openai_format_data)}")
    print(f"已保存至: {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将微信导出 CSV 转为训练用 JSONL（OpenAI 角色字段）")
    parser.add_argument("-i", "--input", default="data/私聊_周珂帆.csv", help="输入 CSV 路径")
    parser.add_argument("-o", "--output", default="data/周珂帆.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--split-minutes", type=int, default=60, help="超过该间隔（分钟）则视为新会话，默认 60")
    args = parser.parse_args()
    build_training_dataset(args.input, args.output, args.split_minutes)
