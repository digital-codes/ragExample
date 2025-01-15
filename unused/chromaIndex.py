import json
import os
import re
import chromadb
#from chromadb.config import Settings
#from chromadb.utils import embedding_functions
import datetime
import sys

basedir = 'tika2024'
dbDir = './chroma/ka-rat-db-1000'
dbCollection = "tika"

chroma_client = chromadb.PersistentClient(path=dbDir)
#chroma_client = chromadb.Client()

try:
    collection = chroma_client.create_collection(name=dbCollection)
    indexData = True
    print("Indexing required")
except chromadb.errors.UniqueConstraintError:
    collection = chroma_client.get_collection(name=dbCollection)
    indexData = False
    print("Collection opened")
    

def preprocess_text(text):
    """
    Preprocess the text by removing extra whitespace, newlines, and non-essential characters.
    
    Args:
        text (str): The raw text to preprocess.
    
    Returns:
        str: The cleaned and preprocessed text.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces, tabs, and newlines with a single space
    text = re.sub(r'\.{2,}', '.', text)  # Replace multiple periods with a single period
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def chunk_text(text, chunk_size=1000, overlap=50):
    """
    Split the text into smaller chunks of a fixed size with an overlap.
    
    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk (in tokens, not characters).
        overlap (int): The number of tokens to overlap between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    words = text.split()  # Split text into words
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def prepare_indexed_data(base, filename):
    indexed_data = []
    file_path = os.sep.join([base, filename])
    if not os.path.exists(file_path):
        raise ValueError(f"File does't exist: {file_path}")

    print(f"Preparing indexed data from folder: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            document = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {file_path}: {e}")
            return []
        
        if "content" not in document:
            print(f"No 'content' field in file {file_path}. Skipping.")
            return []
        
        meta = {"filename": filename,"indexdate": datetime.datetime.now().isoformat()}
        # Extract metadata, if exist
        metakeys = ['title', 'author', 'date', 'url', 'source', 'description', 
                    'keywords',"dc:language","dcterms:modified","dc:language"
                    ]
        if "metadata" in document:
            for key, value in document["metadata"].items():
                if key in metakeys:
                    if isinstance(value, str):
                        meta[key] = value
        
        raw_content = document['content']
        clean_content = preprocess_text(raw_content)
        chunks = chunk_text(clean_content)
        
        for i, chunk in enumerate(chunks):
            indexed_data.append({
                'ids': f"{filename}_chunk_{i}",  # Unique identifier for each chunk
                'content': chunk,
                "metadatas": meta
            })
    
    return indexed_data
    

if indexData:  
    files = os.listdir(basedir)

    for f in files:
        if not f.endswith('.json'):
            continue
        data = prepare_indexed_data(basedir, f)
        try:
            #print(data)
            print(len(data))
            if len(data) == 0:
                continue
            documents = [record['content'] for record in data]
            metadatas = [record['metadatas'] for record in data]
            ids = [record['ids'] for record in data]
            #print(documents,metadatas,ids)
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f)
        except:
            print("Error in adding document")
            continue
        
