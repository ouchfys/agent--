import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from time import sleep
import pdfplumber
import PyPDF2
from PyPDF2 import PdfFileReader
from langchain_core.documents import BaseDocumentTransformer, Document
import numpy
#from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


#base_model = AutoModel.from_pretrained("/home/zy/LLM/chatglm3-6b-32k/", device_map="auto", trust_remote_code=True)
#base_model.to("cuda")
#base_model.gradient_checkpointing_enable()

peft_config = PeftConfig.from_pretrained("./lora_results/checkpoint-100")

model = PeftModel.from_pretrained(
    AutoModel.from_pretrained("/home/zy/LLM/chatglm3-6b-32k/", device_map="auto", trust_remote_code=True),
    #"./lora_results/checkpoint-100",
    "./lora_results/checkpoint-100",
    #adapter_name="my_lora",
    device_map="auto",
    config=peft_config
)

tokenizer = AutoTokenizer.from_pretrained("/home/zy/LLM/chatglm3-6b-32k/", trust_remote_code=True)

#inputs = tokenizer("""
#已知：机壳地、信号地及电源地一般通过搭接耳片连接到接地桩或通过接地模块接地。具体的测试要求见 5.10 部分。
#问题：机壳地、信号地及电源地通过接地桩搭接的测试要求是什么？
#请回答：根据已知内容回答问题，如果已知内容不足够回答问题，则还需要补充哪些内容？
#""", return_tensors="pt").to("cuda")

prompt = """
已知：机壳地、信号地及电源地一般通过搭接耳片连接到接地桩或通过接地模块接地。具体的测试要求见 5.20 部分
问题：机壳地、信号地及电源地通过接地桩搭接的测试要求是什么
请回答：根据已知内容回答问题，如果已知内容不足够回答问题，则还需要补充哪些内容？
"""

# 生成时指定使用哪个适配器
#outputs = model.generate(
#    **inputs,
#    max_new_tokens=200
#)
for response in model.stream_chat(
    tokenizer,
    prompt,
    max_length=200,
    top_p=0.8,
    temperature=0.15,
):
    outputs = response


result = outputs[0]
responses = ""
for i in range(len(result)):
    if result[i] != "\n":
        responses = responses+result[i]
    else:
        break
print(responses)