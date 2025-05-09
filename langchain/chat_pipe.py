# chat_with_rag.py
# initial from https://docling-project.github.io/docling/examples/rag_langchain/#setup

# https://python.langchain.com/docs/tutorials/rag/
# https://python.langchain.com/docs/tutorials/qa_chat_history/

# for improvements: https://python.langchain.com/docs/how_to/qa_sources/
# https://python.langchain.com/docs/how_to/qa_citations/
# https://python.langchain.com/docs/how_to/qa_streaming/

import sys
import os
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
import private_remote as pr 

from langchain_community.chat_models import ChatDeepInfra
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA  # outdated

def setup(storeName, emblocal=False):
    # Initialize your custom embedder
    if not emblocal:
        from langchain_community.embeddings import DeepInfraEmbeddings
        embMdl = "BAAI/bge-m3"
        embedder = DeepInfraEmbeddings(model_id=embMdl, deepinfra_api_token=pr.deepInfra["apiKey"])
    else:
        import localEmbeddings as LE
        embedder = LE.LocalEmbeddings("bge-m3-Q4_K_M")

    # Load FAISS vector store
    print(f"Loading {storeName}...")       
    vector_store = FAISS.load_local(
        storeName, embedder, allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully.")    
    return vector_store

    
def main():
    parser = argparse.ArgumentParser(description="Chat with a FAISS index using RAG.")
    parser.add_argument("storeName", type=str, help="Name of the FAISS index to be used.")
    parser.add_argument("--emblocal", action='store_true', help="Use local embeddings instead of DeepInfra.")
    parser.add_argument("--llmlocal", action='store_true', help="Use local llm instead of DeepInfra.")
    args = parser.parse_args()
    storeName = args.storeName
    emblocal = args.emblocal
    llmlocal = args.llmlocal

    vectorstore = setup(storeName, emblocal)

    if not llmlocal:
        # Setup DeepInfra LLM
        llm = ChatDeepInfra(model="meta-llama/Llama-3.3-70B-Instruct-Turbo",deepinfra_api_token=pr.deepInfra["apiKey"])
    else:
        import localChatModel as LC
        llm = LC.ChatLocal(parrot_buffer_length=3, model="my_custom_model")
        print("Using local model",llm)
        
    # Step 4: Build RAG chain
    retriever = vectorstore.as_retriever(search_type="mmr", #similarity_score_threshold", 
                                         search_kwargs={"k": 5,"score_threshold": 0.4})
    
    # test retriever
    # docs = retriever.invoke("what does docling deliver")
    # print("Retrieved documents:")
    # for doc in docs:
    #    print(doc)
    
    # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
    #print("Prompt loaded successfully.",prompt)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


    # Step 5: Chat loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() in ("exit", "quit"):
            break

        result = rag_chain.invoke({"query": query})
        #result = rag_chain.invoke({"input": query})

        print("\nAnswer:")
        print(result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")


if __name__ == "__main__":
    main()
#