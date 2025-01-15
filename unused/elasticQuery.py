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
   
def queryByText(collection, model, query):
    embedding = model.encode(query)
    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": [
                    {
                        "script_score": {
                            "query": {
                                "match_all": {}
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": embedding}
                            }
                        }
                    }
                ]
            }
        }
    }
    response = es.search(index=collection, body=search_query)
    result = []
    # results ordered by score automatically
    for hit in response['hits']['hits']:
        #print(f"Score: {hit['_score']}, Content: {hit['_source']['content']}")
        result.append({
            "scrore:":hit['_score'],
            "content":hit['_source']['content'],
            "metadata": hit['_source']['metadata'],
            "title":hit['_source']['document_title'],
            "file":hit['_source']['document_id'],
            "type":hit['_source']['document_type'],
            "id":hit['_source']['chunk_id']
            })
    
    return result



r = checkIndex(dbCollection)
if not r:
    print("Index does not exist")
    sys.exit(1)

model = create_embedder(embedder)

## input 
query = input("Enter your query: ")

results = queryByText(dbCollection,model, query)


print(results)

