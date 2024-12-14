import os
import openai
import sys
from elasticsearch import Elasticsearch, BadRequestError
from sentence_transformers import SentenceTransformer, models

debug = False

dbCollection = "tika2024"

embedderModels = ["deepset/gbert-base"]
embedderModel = embedderModels[0]

llmModels = ["o1-mini", "gpt-4o-mini", "gpt-4o"]
llm = llmModels[0]

try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError
except:
    print("OpenAI API key not found")
    sys.exit()


es = Elasticsearch(hosts = 'http://localhost:9200')
if es.ping():
    print('Connected to ES!')
else:
    print('Could not connect!')
    sys.exit(1)

if not es.indices.exists(index=dbCollection):
    print("Index does not exist")
    sys.exit(1)
        

def create_embedder(modelName = embedderModels[0]):
    # maybe check if model is directly available as sentence-transformers model
    # ... or use Huggingface model directly
    # Load GBERT model as a Huggingface transformer
    word_embedding_model = models.Transformer(modelName)
    # Add a pooling layer to convert hidden states to embeddings
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    # Create a full SentenceTransformer model using Huggingface + Pooling
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

embedder = create_embedder(embedderModel)


# Function to query ElasticSearch for relevant context   
def queryByText(collection, model, query, top_n=3):
    embedding = model.encode(query)
    search_query = {
        "size": top_n,
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
    results = []
    # results ordered by score automatically
    for idx,hit in enumerate(response['hits']['hits']):
        #print(f"Score: {hit['_score']}, Content: {hit['_source']['content']}")
        results.append({
            "idx":idx,
            "score":hit['_score'],
            "content":hit['_source']['content'],
            "metadata": hit['_source']['metadata'],
            "title":hit['_source']['document_title'],
            "file":hit['_source']['document_id'],
            "type":hit['_source']['document_type'],
            "id":hit['_source']['chunk_id']
            })
    if debug: print(results)
    context = "\n".join([r['content'] for r in results])
    # document_index doesn't help
    # context_with_id = "\n".join([f"Document Index {r['idx']+1}:\n{r['content']}\n" for r in results])
    ids = [r['id'] for r in results]
    meta = [r["metadata"] for r in results]
    scores = [r["score"] for r in results]
    return context, ids, scores, meta


# Function to call OpenAI API with context and query
def query_openai_with_context(context, query):
    """Send the context and query to OpenAI's API and return the response."""
    prompt = f"""
    You are an intelligent assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}
    
    Provide a clear, concise answer in German language. Limit your response to 100 words.
    """
    if debug: print(prompt)
    response = openai.chat.completions.create(
        model=llm,
        messages=[
            #{"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}
        ]
    )
    return response
    

# End-to-End Workflow
def rag_system(query):
    """Main workflow for the RAG system."""
    if debug: print("\n🔍 Searching for relevant context in ChromaDB...")
    # context, ids, distances, meta = search_chromadb(query, top_n=3)
    context, ids, scores, meta =  queryByText(dbCollection, embedder, query, top_n=3)
    
    if debug: print(f"\n📚 Context found:\n{context}\n")
    
    answer = query_openai_with_context(context, query)
    if debug: print(f"\n💡 OpenAI's response:\n{answer}\n")
    return answer, ids, scores, meta

## ####### options ######
import pandas as pd

def load_csv_rows(file_path, column_filters=None, row_limit=3):
    """Loads a limited number of rows from a CSV."""
    df = pd.read_csv(file_path)
    if column_filters:
        df = df[column_filters]
    return df.head(row_limit)




# Run the RAG system
if __name__ == "__main__":
    query = input("\n🔍 Enter your query: ")
    answer, ids, scores, meta = rag_system(query)
    result = answer.choices[0].message.content.strip()
    print(result)
    print(ids)
    print(scores)
    print(meta)
    
