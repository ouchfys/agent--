from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
import requests
#修改的地方-开始
import server
#修改的地方-截至

#model_name = "C:\\Users\\33121\\Desktop\\command line\\bge-large-zh-v1.5"

# 修改的地方-开始
#model_name = "/home/zy/embaddingModel/bge-large-zh-v1.5"
# 修改的地方-截至

# encode_kwargs = {'normalize_embeddings': True}
# embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     encode_kwargs=encode_kwargs,
# )

# 修改的地方-开始
model_name = "/home/zy/embaddingModel/bge-large-zh-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
# 修改的地方-截至

url = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"
neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embedding,
    node_label="Chunk",
    embedding_node_property="embedding",
    text_node_properties=["text"],
    url=url,
    username=username,
    password=password
)
    
contexts = ""
user_question = ""

def llm_answer(question):
    global contexts
    k = 1
    results = neo4j_vector.similarity_search(question, k=k)
    for i in range(k):
        contexts += f"相关内容{i+2}:" + results[i].page_content + '\n\n'
    source = f"已知：\n\n{contexts} \n\n。回答问题：{user_question}\n\n"
    from langchain.prompts import PromptTemplate
    template = """
特别注意，只能根据所提供的相关内容来回答问题。
{material}
            """
    prompt = PromptTemplate(
        input_variables=["material"],
        template=template,
    )
    prompt = prompt.format(material=source)
    # return requests.post(
    #     'http://192.168.8.84:5000/question_to_answer',
    #     json={'question': prompt}
    # )
    return server.glm3_answer(prompt)
    
def lora_llm_answer(question):
    global contexts
    user_question = question
    k = 1
    results = neo4j_vector.similarity_search(question, k=k)
    for i in range(k):
        contexts += f"相关内容{i+1}:" + results[i].page_content + '\n\n'
    source = f"已知：{contexts} 。问题：{question}\n"
    from langchain.prompts import PromptTemplate
    template = """
{material}
请回答：根据已知内容回答问题，如果已知内容不足够回答问题，则还需要补充哪些内容？
            """
    prompt = PromptTemplate(
        input_variables=["material"],
        template=template,
    )
    prompt = prompt.format(material=source)
    # return requests.post(
    #     'http://192.168.8.84:5000/question_to_answer',
    #     json={'question': prompt}
    # )
    return server.lora_glm3_answer(prompt)

def initial_contexts():
    global contexts
    contexts = ""