import sys
import os

# 将 system 目录加入路径，这样才能导入 file_extraction 和 data_import
sys.path.append(os.path.join(os.path.dirname(__file__), 'system'))

from file_extraction import extract_text_from_pdf
from data_import import write_segments_to_db

# 配置你的论文路径
PDF_PATH = "/root/autodl-tmp/data/纯论文-面向工业文档知识问答的多轮检索增强生成方法研究(1).pdf"

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(PDF_PATH):
        print(f"错误：找不到文件 {PDF_PATH}")
        print("请检查文件是否已上传到 /root/autodl-tmp/data/ 目录下")
        return

    print(f">>> 开始处理文件: {os.path.basename(PDF_PATH)}")

    # 2. 提取文本 (调用 system/file_extraction.py)
    print(">>> 正在提取文本和预处理...")
    # 注意：extract_text_from_pdf 里面有一些针对特定水印的硬编码去除逻辑(rHg等)，
    # 针对你的新论文可能不完全适用，但咱们先跑通流程。
    documents = extract_text_from_pdf(PDF_PATH)
    print(f">>> 提取完成，共生成 {len(documents)} 个文本块")

    # 3. 写入数据库 (调用 system/data_import.py)
    if len(documents) > 0:
        print(">>> 正在写入 Neo4j 数据库 (这一步需要加载 Embedding 模型，可能稍慢)...")
        file_name = os.path.basename(PDF_PATH)
        try:
            write_segments_to_db(documents, file_name)
            print(">>> ✅ 成功！论文已入库 Neo4j。")
        except Exception as e:
            print(f">>> ❌ 写入数据库失败: {e}")
            print("请检查：1. Neo4j是否已启动？ 2. 密码是否正确？ 3. model_name路径是否正确？")
    else:
        print(">>> ⚠️ 警告：没有提取到任何文本，请检查PDF是否加密或全是图片。")

if __name__ == "__main__":
    main()