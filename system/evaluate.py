import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch

# 添加 system 目录以导入 file_extraction
sys.path.append(os.path.join(os.path.dirname(__file__), 'system'))
try:
    from file_extraction import extract_text_from_pdf
except ImportError:
    # 尝试直接导入
    from file_extraction import extract_text_from_pdf

# ================= 1. 配置区域 (请填入你的 DeepSeek Key) =================
# ⚠️⚠️⚠️在此处填入你的 DeepSeek API Key ⚠️⚠️⚠️
DEEPSEEK_API_KEY = "sk-cc62600607034908acd7e755ffef5e66" 

# 本地模型路径
MODEL_PATH = "/root/autodl-tmp/models/chatglm3-6b"
EMBEDDING_PATH = "/root/autodl-tmp/models/bge-large-zh-v1.5"

# Neo4j 配置
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# ================= 2. 初始化模型 =================

# --- A. 考官 (DeepSeek) ---
print("👨‍⚖️ 正在初始化 DeepSeek 考官...")
judge_llm = ChatOpenAI(
    model="deepseek-chat",            # DeepSeek V3
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com", # DeepSeek 官方接口地址
    temperature=0 # 评估时保持冷静，不要随机
)

# --- B. 考生 (本地 ChatGLM3) ---
class ChatGLM3Wrapper:
    """简单的 ChatGLM3 包装器，用于生成回答"""
    def __init__(self):
        print("⏳ 正在加载本地 ChatGLM3 (考生)...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype="auto").eval()

    def invoke(self, prompt: str) -> str:
        response, _ = self.model.chat(self.tokenizer, prompt, history=[], do_sample=False)
        return response

local_llm = ChatGLM3Wrapper()

# --- C. 向量检索 (Neo4j) ---
print("⏳ 正在连接 Neo4j 知识库...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_PATH,
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="vector",
    node_label="Chunk",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

# ================= 3. 生成测试数据 =================
def generate_test_data_with_deepseek(pdf_path, num=3):
    """
    让 DeepSeek 阅读论文片段，并生成高质量的问题和标准答案 (Ground Truth)
    """
    print(f"📄 正在读取论文: {os.path.basename(pdf_path)}")
    chunks = extract_text_from_pdf(pdf_path)
    
    if not chunks:
        print("❌ 未提取到文本，请检查 file_extraction.py")
        return []

    import random
    # 随机选几个片段
    selected_chunks = random.sample(chunks, min(len(chunks), num))
    
    test_set = []
    print("🧠 DeepSeek 正在出题...")
    
    for chunk in selected_chunks:
        context = chunk.page_content
        
        # 让 DeepSeek 出题
        prompt = f"""
        你是一个专业的考官。请根据以下技术文档片段，生成一个具体的、有深度的问题，并根据文档内容给出标准答案。
        
        【文档片段】：
        {context}
        
        请严格按以下格式输出（不要输出其他废话）：
        问题：xxx
        答案：xxx
        """
        
        response = judge_llm.invoke(prompt).content
        
        # 简单的解析逻辑
        try:
            q_part = response.split("问题：")[1].split("答案：")[0].strip()
            a_part = response.split("答案：")[1].strip()
            
            test_set.append({
                "question": q_part,
                "ground_truth": a_part
            })
            print(f"  ✅ 生成题目: {q_part[:20]}...")
        except:
            print("  ⚠️ 解析题目失败，跳过一条")
            
    return test_set

# ================= 4. 运行评估 =================
def run_evaluation(test_data):
    print("\n🚀 开始 RAG 考试 (Local ChatGLM3 作答)...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        # 1. 检索
        docs = vector_store.similarity_search(q, k=3)
        retrieved_text = [d.page_content for d in docs]
        
        # 2. 本地模型生成回答
        context_str = "\n".join(retrieved_text)
        prompt = f"基于已知信息：\n{context_str}\n\n问题：{q}"
        ans = local_llm.invoke(prompt)
        
        questions.append(q)
        answers.append(ans)
        contexts.append(retrieved_text)
        ground_truths.append(gt) # Ragas 需要 list of strings for GT? No, usually just string in list
        
    # 构建数据集
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    print("\n👨‍⚖️ DeepSeek 正在阅卷 (运行 Ragas 指标)...")
    # 关键点：把 judge_llm 传给 Ragas
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_recall,
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        llm=judge_llm,       # <--- DeepSeek 做裁判
        embeddings=embedding_model # 依然用本地 BGE 做向量计算
    )
    
    return results

if __name__ == "__main__":
    # 默认路径
    default_pdf = "/root/autodl-tmp/agent项目/data/论文（无英文文献）.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = default_pdf
    
    # 1. 出题
    test_data = generate_test_data_with_deepseek(pdf_path, num=3) # 演示用3条，正式可以用10条
    
    if test_data:
        # 2. 考试 & 阅卷
        results = run_evaluation(test_data)
        
        print("\n🏆 评估报告:")
        print(results)
        
        # 保存
        df = results.to_pandas()
        df.to_csv("deepseek_eval_report.csv", index=False)
        print("✅ 结果已保存至 deepseek_eval_report.csv")
    else:
        print("❌ 未生成测试数据")