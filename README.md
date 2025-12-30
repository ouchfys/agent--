# 🏭 Industrial Graph RAG Agent | 工业文档智能问答系统

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green) ![ChatGLM3](https://img.shields.io/badge/LLM-ChatGLM3--6B-orange) ![RAG](https://img.shields.io/badge/RAG-Graph%20Augmented-purple)

> 一个基于 **ChatGLM3-6B** + **Neo4j** 的垂直领域知识问答系统。
> 🚀 **核心特性**：结构化文档解析 (Data-Centric) | 知识图谱增强检索 (Graph RAG) | 自动化量化评估 (MLOps)

## 📖 项目背景 (Background)

传统的 RAG（检索增强生成）系统在处理长篇工业/学术文档时，常面临**“上下文丢失”**和**“多跳推理困难”**的问题。
本项目放弃了简单的文本切片策略，采用 **Data-Centric AI** 思维，通过深度解析 PDF 文档结构，构建**文本-实体-关系**混合索引的知识图谱，显著提升了复杂问题的召回准确率和回答的忠实度。

---

## 🌟 核心亮点 (Key Features)

### 1. Data-Centric 文档治理 (Structured Parsing)
不同于传统的 `Chunking`，本项目实现了基于文档层级的解析算法：
- **层级元数据提取**：自动识别 PDF 中的章、节、页码信息，实现 **Parent-Child Indexing**（父子索引）。
- **结构化清洗**：利用 `pdfplumber` 精准提取文本，去除页眉页脚干扰。

### 2. Graph RAG 图谱增强 (Knowledge Graph)
利用 **Neo4j** 图数据库超越单纯的向量检索：
- **混合索引**：结合 Vector Search（语义检索）与 Graph Traversal（图遍历）。
- **实体对齐**：通过 `jieba` 提取关键词构建 `(:Chunk)-[:HAS_KEYWORD]->(:Keyword)` 关系，解决专有名词检索难题。
- **上下文链表**：建立 `(:Chunk)-[:NEXT_CHUNK]->(:Chunk)` 关系，检索时自动回溯上下文，保证回答连贯性。

### 3. MLOps 自动化评估流水线 (Automated Evaluation)
拒绝“凭感觉”调优，构建了基于 **DeepSeek/GPT-4** 的 LLM-as-a-Judge 评估体系：
- **自合成数据**：自动从文档生成 (Question, Ground_Truth) 测试集。
- **量化指标**：计算 **Context Recall (召回率)** 和 **Faithfulness (忠实度)**，数据驱动系统迭代。

---

## 🛠️ 技术栈 (Tech Stack)

- **LLM (Generator)**: ChatGLM3-6B (Local Deployment)
- **Embedding**: BAAI/bge-large-zh-v1.5
- **Database**: Neo4j (Vector + Graph)
- **Orchestration**: LangChain, Py2Neo
- **Frontend**: Streamlit
- **Evaluation**: DeepSeek API / Ragas / Custom Pipeline
- **Tools**: AutoDL, CUDA, PyTorch

---

## 📂 项目结构 (Structure)

```text
.
├── system/
│   ├── data_import.py      # [Core] 图谱构建：向量入库 + 实体关系建立
│   ├── file_extraction.py  # [Core] 数据治理：PDF 结构化解析与清洗
│   └── evaluate.py         # [Core] 评估流水线：DeepSeek 自动化打分
├── web_demo_streamlit_3.py # [App] 前端交互界面 (RAG 逻辑集成)
├── run_ingest.py           # 数据入库入口脚本
├── requirements.txt        # 依赖清单
├── README.md               # 项目文档
└── data/                   # 存放原始 PDF 文档
