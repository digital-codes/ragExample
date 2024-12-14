import os
import openai
import chromadb
# use default embeddings
#from chromadb.utils import embedding_functions
import sys

debug = False

models = ["o1-mini", "gpt-4o-mini", "gpt-4o"]
model = models[0]


try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError
except:
    print("OpenAI API key not found")
    sys.exit()

# Step 1: Initialize ChromaDB client and collection
# assume document collection is already created
basedir = 'tika2024'
dbDir = './chroma/ka-rat-db-200'
dbCollection = "tika"

chroma_client = chromadb.PersistentClient(path=dbDir)

try:
    collection = chroma_client.get_collection(name=dbCollection)
except:
    print("Collection not found")
    sys.exit()
    
# Step 2: Function to search ChromaDB for relevant context
def search_chromadb(query, top_n=3):
    """Search for top_n most relevant context passages from ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_n,
    )
    print(results)
    context = "\n".join([doc for doc in results['documents'][0]])
    return context, results["ids"], results["distances"], results["metadatas"]

# Step 3: Function to call OpenAI API with context and query
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
        model=model,
        messages=[
            #{"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}
        ]
    )
    return response
    

# Step 4: End-to-End Workflow
def rag_system(query):
    """Main workflow for the RAG system."""
    if debug: print("\n🔍 Searching for relevant context in ChromaDB...")
    context, ids, distances, meta = search_chromadb(query, top_n=3)
    if debug: print(f"\n📚 Context found:\n{context}\n")
    
    answer = query_openai_with_context(context, query)
    if debug: print(f"\n💡 OpenAI's response:\n{answer}\n")
    return answer, ids, distances, meta

## ####### options ######
import pandas as pd

def load_csv_rows(file_path, column_filters=None, row_limit=3):
    """Loads a limited number of rows from a CSV."""
    df = pd.read_csv(file_path)
    if column_filters:
        df = df[column_filters]
    return df.head(row_limit)

def query_with_csv_context(query):
    # Search for chunks in ChromaDB
    results = collection.query(query_texts=[query], n_results=3)
    combined_context = ""

    for i, chunk in enumerate(results['documents'][0]):
        context = chunk
        metadata = results['metadatas'][0][i]
        
        # Check if CSV file is associated
        if 'file_name' in metadata:
            file_path = f"./data/{metadata['file_name']}"
            columns = metadata.get('columns', None)
            csv_rows = load_csv_rows(file_path, column_filters=columns, row_limit=3)
            
            # we can add row names to the context
            # or additional table content, like the first 3 rows
            # row names can be retrieved dynamically from the CSV file like in this code
            # and/or stored with the metadata in the database
            
            # Add CSV rows to context
            csv_text = csv_rows.to_markdown(index=False)
            context += f"\n\nRelevant Table from {metadata['file_name']}:\n{csv_text}"

            
        combined_context += context + "\n\n"

    return combined_context




# Step 5: Run the RAG system
if __name__ == "__main__":
    query = input("\n🔍 Enter your query: ")
    answer, ids, distances, meta = rag_system(query)
    result = answer.choices[0].message.content.strip()
    print(result)
    print(ids)
    print(distances)
    print(meta)
    
