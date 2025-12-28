import os
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# ================= 1. 路径配置 =================
MODEL_PATH = "/root/autodl-tmp/models/chatglm3-6b"
LORA_PATH = "/root/autodl-tmp/agent项目/lora_results"

print("="*50)
print("🚀 开始终端 LoRA 完整性测试")
print("="*50)

# ================= 2. 加载 Base Model =================
print(f"⏳ 正在加载基座模型: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype="auto" # 满血 FP16
    ).eval()
    print("✅ 基座模型加载成功")
except Exception as e:
    print(f"❌ 基座模型加载失败: {e}")
    exit(1)

# ================= 3. 加载 LoRA =================
print(f"⏳ 正在加载 LoRA 权重: {LORA_PATH}")
try:
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA 权重加载成功！")
except Exception as e:
    print(f"❌ LoRA 加载严重失败 (文件可能损坏): {e}")
    exit(1)

# ================= 4. 生成测试 =================
query = "什么是工业文档知识问答？"
print("-" * 30)
print(f"❓ 测试问题: {query}")
print("-" * 30)

# 测试 A: 贪婪搜索 (Greedy Search) - 最稳
print("🧪 [测试 A] 贪婪搜索 (do_sample=False, repetition_penalty=1.2)")
try:
    response, history = model.chat(
        tokenizer, 
        query, 
        history=[], 
        do_sample=False,          # 关键：不采样
        repetition_penalty=1.2    # 关键：防复读
    )
    print(f"🤖 回答:\n{response}")
except Exception as e:
    print(f"❌ 测试 A 报错: {e}")

print("-" * 30)

# 测试 B: 默认参数 (Default) - 模拟 Streamlit 默认行为
print("🧪 [测试 B] 默认参数 (do_sample=True, top_p=0.8)")
try:
    response, history = model.chat(
        tokenizer, 
        query, 
        history=[], 
        do_sample=True,
        top_p=0.8,
        temperature=0.8
    )
    print(f"🤖 回答:\n{response}")
except Exception as e:
    print(f"❌ 测试 B 报错 (可能是 NaN): {e}")

print("="*50)
print("测试结束。")