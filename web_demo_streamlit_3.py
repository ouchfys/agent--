import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
# from peft import PeftModel # LoRA removed
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.vectorstores import Neo4jVector
except ImportError:
    from langchain.vectorstores import Neo4jVector

# ================= 配置路径 =================
# 请确保这些路径与你 AutoDL 上的实际路径一致
# 如果在本地运行，请修改为本地路径
MODEL_PATH = "/root/autodl-tmp/models/chatglm3-6b"
EMBEDDING_PATH = "/root/autodl-tmp/models/bge-large-zh-v1.5"

# 设置页面
st.set_page_config(page_title="工业文档知识问答 Agent", page_icon="🤖", layout="wide")
st.title("🤖 工业文档知识问答 Agent (Base Model + RAG)")
st.markdown("### 🚀 基于 ChatGLM3-6B (无 LoRA) 与 Neo4j 知识图谱")

# ================= 1. 加载模型 =================
@st.cache_resource
def load_models():
    print("⏳ [System] 正在加载 Embedding 模型...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_PATH,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    except Exception as e:
        st.error(f"Embedding 模型加载失败: {e}")
        return None, None, None
    
    print("⏳ [System] 正在连接 Neo4j...")
    try:
        # 此时连接 Neo4j，注意 index_name 必须与 data_import.py 中一致
        vector_store = Neo4jVector.from_existing_graph(
            embedding=embedding,
            url="bolt://localhost:7687",
            username="neo4j",
            password="12345678",
            index_name="vector",
            node_label="Chunk",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
    except Exception as e:
        st.error(f"Neo4j 连接失败: {e}")
        return None, None, None

    print("⏳ [System] 正在加载 ChatGLM3 模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # 满血 FP16 加载，适合 3090/4090 显卡
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype="auto").eval()
    except Exception as e:
        st.error(f"ChatGLM3 模型加载失败: {e}")
        return None, None, None

    # LoRA 加载逻辑已移除
    
    return tokenizer, model, vector_store

tokenizer, model, vector_store = load_models()

# ================= 2. 状态管理 =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= 3. 侧边栏与功能 =================
with st.sidebar:
    st.header("⚙️ 控制面板")
    # 强制清空按钮：解决乱码的关键
    if st.button("🗑️ 清空对话历史"):
        st.session_state.history = []
        st.rerun()
    st.info("💡 如果回复出现乱码，请点击上方“清空对话历史”按钮。")

# 显示历史消息
for query, response in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(response)

# ================= 4. 核心问答逻辑 =================
if prompt_text := st.chat_input("请输入您的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # RAG 检索
    context_str = ""
    print(f"🔍 用户提问: {prompt_text}")
    try:
        with st.status("🔍 正在检索知识库 (Neo4j)...", expanded=False) as status:
            if vector_store:
                docs = vector_store.similarity_search(prompt_text, k=3)
                if docs:
                    context_str_list = []
                    status.write("✅ 检索到相关知识片段：")
                    for i, d in enumerate(docs):
                        # 获取元数据
                        source = d.metadata.get('source', '未知文件')
                        chapter = d.metadata.get('chapter', '未知章节')
                        page = d.metadata.get('page', '?')
                        
                        source_info = f"【来源】{source} - {chapter} (P{page})"
                        status.markdown(f"**片段 {i+1}:** {source_info}")
                        status.code(d.page_content[:200] + "...", language="text")
                        
                        context_str_list.append(f"Content: {d.page_content}\nSource: {source_info}")
                    
                    context_str = "\n\n".join(context_str_list)
                else:
                    status.write("⚠️ 未检索到直接相关内容，将尝试使用模型通用知识回答。")
                status.update(label="检索完成", state="complete", expanded=False)
            else:
                status.write("⚠️ 向量库未连接，仅使用模型回答。")
                status.update(label="检索跳过", state="error", expanded=False)
                
    except Exception as e:
        st.error(f"检索出错: {e}")

    # 构造 Prompt
    input_prompt = f"基于以下已知信息，简洁专业地回答用户的问题。如果不确定，请说明。\n\n【已知信息】\n{context_str}\n\n【用户问题】\n{prompt_text}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        if model:
            try:
                # 关键参数设置：防止乱码和报错
                for response, history, past_key_values in model.stream_chat(
                    tokenizer, 
                    input_prompt, 
                    history=st.session_state.history, 
                    do_sample=False,        # 关闭随机采样，解决 NaN 报错
                    repetition_penalty=1.2, # 核心：惩罚重复，解决方块字乱码
                    max_length=4096,
                    past_key_values=None,
                    return_past_key_values=True
                ):
                    placeholder.markdown(response)
                    full_response = response
                
                st.session_state.history = history
                
            except Exception as e:
                st.error(f"生成出错: {e}")
                print(f"❌ 生成出错: {e}")
        else:
            st.error("模型未加载，无法生成回答。")
