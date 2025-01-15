import json
import os
import re
import datetime
import sys

from elasticsearch import Elasticsearch, BadRequestError
from sentence_transformers import SentenceTransformer, models

basedir = 'tika2024'

dbCollection = "tika2024"
#elurl = f"http://localhost:9200"

es = Elasticsearch(hosts = 'http://localhost:9200')
if es.ping():
    print('Connected to ES!')
else:
    print('Could not connect!')
    sys.exit(1)


embedder = "deepset/gbert-base"

schema = {
  "mappings": {
    "properties": {
      "chunk_id": {
        "type": "keyword"
      },
      "document_id": {
        "type": "keyword"
      },
      "document_title": {
        "type": "keyword"
      },
      "document_type": {
        "type": "keyword"
      },
      "content": {
        "type": "text"
      },
      "metadata": {
        "type": "object"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 768
      }
    }
  }
}

def createIndex(name,schema):
    try:
        response = es.indices.create(index=name, body=schema)
        print(f"Index '{name}' created successfully with response:", response)
    except BadRequestError:
        print(f"Error creating index '{name}':")
        sys.exit()
    except:
        print("Error in creating index")
        sys.exit()


def checkIndex(name):
    if es.indices.exists(index=name):
        print("Index exists")
        return True
    else:
        print("Index does not exist")
        return False
        

def create_embedder(modelName = embedder):
    # Step 1: Load GBERT model as a Huggingface transformer
    word_embedding_model = models.Transformer(modelName)

    # Step 2: Add a pooling layer to convert hidden states to embeddings
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)

    # Step 3: Create a full SentenceTransformer model using Huggingface + Pooling
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model
   

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
    



r = checkIndex(dbCollection)
if not r:
    print("Creating index")
    createIndex(dbCollection,schema)

model = create_embedder(embedder)
files = os.listdir(basedir)
    

for f in files:
    if not f.endswith('.json'):
        continue
    print(f)
    data = prepare_indexed_data(basedir, f) # return list of chunks
    try:
        #print(data)
        print(len(data))
        if len(data) == 0:
            continue
        for chunk in data:
            #print(chunk)
            chunkId = chunk["ids"]
            print(chunkId)
            meta = chunk["metadatas"]
            embedding = model.encode(chunk["content"])
            #print(embedding)
            doc = {
                    'chunk_id': chunkId,
                    'document_id': f,
                    'document_title': meta.get('title', 'Untitled'),
                    'document_type': meta.get('type', 'PDF'),
                    'content': chunk["content"],
                    'metadata': chunk["metadatas"],
                    'embedding': list(embedding)
                }
            es.index(index=dbCollection, id=chunkId, body=doc)
    except:
        print("Error in adding document")
        continue
    
