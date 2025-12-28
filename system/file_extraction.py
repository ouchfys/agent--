import pdfplumber
import re
import os
from langchain_core.documents import Document

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber, identifying chapters and pages.
    Returns a list of LangChain Document objects with metadata.
    """
    print(f"Starting extraction for: {pdf_path}")
    
    # Regex for chapter title: starts with "第", followed by anything, then "章"
    # e.g., "第一章 总则"
    chapter_pattern = re.compile(r'^第.*章')
    
    full_text_content = [] # List of {'text': str, 'page': int, 'chapter': str}
    current_chapter = "前言/其他" # Default chapter
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for chapter title
                    # specific logic might be needed depending on PDF format
                    if chapter_pattern.match(line):
                        current_chapter = line
                        print(f"Found Chapter: {current_chapter}")
                    
                    full_text_content.append({
                        'text': line,
                        'page': i + 1,
                        'chapter': current_chapter
                    })
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

    # Group lines into chunks
    chunks = []
    current_chunk_text = ""
    current_chunk_start_page = 1
    current_chunk_chapter = full_text_content[0]['chapter'] if full_text_content else "前言/其他"
    
    CHUNK_SIZE = 500 # Target chunk size
    
    for item in full_text_content:
        # If chapter changes, force a new chunk to avoid cross-chapter chunks
        if item['chapter'] != current_chunk_chapter:
            if current_chunk_text:
                chunks.append(Document(
                    page_content=current_chunk_text,
                    metadata={
                        'source': os.path.basename(pdf_path),
                        'page': current_chunk_start_page,
                        'chapter': current_chunk_chapter
                    }
                ))
            current_chunk_text = item['text']
            current_chunk_start_page = item['page']
            current_chunk_chapter = item['chapter']
        else:
            if len(current_chunk_text) + len(item['text']) > CHUNK_SIZE:
                chunks.append(Document(
                    page_content=current_chunk_text,
                    metadata={
                        'source': os.path.basename(pdf_path),
                        'page': current_chunk_start_page,
                        'chapter': current_chunk_chapter
                    }
                ))
                # Start new chunk
                current_chunk_text = item['text'] 
                current_chunk_start_page = item['page'] 
            else:
                current_chunk_text += "\n" + item['text']
    
    # Add last chunk
    if current_chunk_text:
        chunks.append(Document(
            page_content=current_chunk_text,
            metadata={
                'source': os.path.basename(pdf_path),
                'page': current_chunk_start_page,
                'chapter': current_chunk_chapter
            }
        ))
        
    print(f"Extraction complete. Generated {len(chunks)} chunks.")
    return chunks

def extract_table_from_pdf(pdf_path):
    table_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_data.extend(table)
    return table_data

if __name__ == "__main__":
    # Test path
    pdf_path = "e:\\Agent-learn\\agent项目\\agent项目\\train.csv" # Just a placeholder path, likely won't work as pdf
    # Update to a real PDF if available or mock it
    print("Testing file extraction...")
