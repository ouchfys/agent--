import sys
import os
import jieba.analyse
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from py2neo import Graph
from file_extraction import extract_text_from_pdf

# Configuration
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
# Update model path to be flexible or use a standard one
# Using a local path if it exists, otherwise assuming it might be available or fallback
LOCAL_MODEL_PATH = "/root/autodl-tmp/models/bge-large-zh-v1.5"
HF_MODEL_NAME = "BAAI/bge-large-zh-v1.5"

def get_embedding_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Using local embedding model at {LOCAL_MODEL_PATH}")
        return HuggingFaceBgeEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        print(f"Local model not found, using HuggingFace Hub model: {HF_MODEL_NAME}")
        return HuggingFaceBgeEmbeddings(
            model_name=HF_MODEL_NAME,
            encode_kwargs={'normalize_embeddings': True}
        )

def import_pdf_to_neo4j(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # 1. Extract Data
    print(f"Extracting text from {pdf_path}...")
    chunks = extract_text_from_pdf(pdf_path)
    if not chunks:
        print("No chunks extracted.")
        return

    # 2. Connect to Neo4j for cleanup and post-processing
    try:
        graph = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Delete existing chunks for this file to avoid duplicates
        file_name = os.path.basename(pdf_path)
        print(f"Cleaning up existing data for {file_name}...")
        graph.run("MATCH (c:Chunk {source: $file_name}) DETACH DELETE c", file_name=file_name)
    except Exception as e:
        print(f"Failed to connect to Neo4j or clean data: {e}")
        return

    # 3. Create Vector Store and Chunks
    print("Initializing Embeddings and Vector Store...")
    embedding = get_embedding_model()
    
    try:
        # This will create nodes with label "Chunk" and properties from metadata
        neo4j_vector = Neo4jVector.from_documents(
            chunks,
            embedding,
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="vector",
            node_label="Chunk"
        )
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return

    # 4. Post-processing: Add Keywords and NEXT_CHUNK relationships
    print("Adding Keywords and Relationships...")
    
    for i, chunk in enumerate(chunks):
        # Extract Keywords
        keywords = jieba.analyse.extract_tags(chunk.page_content, topK=5)
        
        # Cypher query to add keywords
        # Match the specific chunk we just added. 
        # Note: We rely on text content being unique enough or the combination of properties.
        # Ideally we'd use the ID returned by Neo4j, but Neo4jVector doesn't return them easily in batch.
        
        chunk_text = chunk.page_content
        
        # Add Keywords
        for kw in keywords:
            query = """
            MATCH (c:Chunk {text: $text, source: $source})
            MERGE (k:Keyword {name: $keyword})
            MERGE (c)-[:HAS_KEYWORD]->(k)
            """
            graph.run(query, text=chunk_text, source=file_name, keyword=kw)
            
        # Add Next Chunk Relationship
        if i < len(chunks) - 1:
            next_chunk_text = chunks[i+1].page_content
            query_rel = """
            MATCH (c1:Chunk {text: $text1, source: $source})
            MATCH (c2:Chunk {text: $text2, source: $source})
            MERGE (c1)-[:NEXT_CHUNK]->(c2)
            """
            graph.run(query_rel, text1=chunk_text, text2=next_chunk_text, source=file_name)

    print(f"Import complete for {file_name}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_import.py <pdf_path>")
    else:
        import_pdf_to_neo4j(sys.argv[1])
