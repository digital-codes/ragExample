# create_index.py
import sys
import os
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
import private_remote as pr 

SAVE_DOCS = True

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

def loadDocs(docpath):
    try:
        from docling.langchain import DoclingLangChainLoader
        loader = DoclingLangChainLoader(docpath) 
        documents = loader.load()
        doc_ids = [f"doc_{i:06}" for i in range(len(documents))]

    except ImportError:
        import json
        with open(docpath, "r", encoding="utf-8") as f:
            docling_data = json.load(f)

        # Check if the data is a list of dictionaries
        if not isinstance(docling_data, list):
            docling_data = [docling_data]  # Wrap in a list if it's not
        #print("Docling data loaded successfully.",docling_data)

        # chunk
        text_splitter = RecursiveCharacterTextSplitter(separators = [".",". ","# ","## ","\n","\n\n"], keep_separator = True, chunk_size=1000, chunk_overlap=200)
        # Split documents into chunks
        documents = []
        doc_ids = []
        for doc_num, entry in enumerate(docling_data):
            content = entry.get("text") or entry.get("content") or ""
            metadata = entry.get("metadata", {})
            splits = text_splitter.split_documents([Document(page_content=content,metadata=metadata)])
            # print("Splits",splits)
            for chunk_num, split in enumerate(splits):
                docId = f"doc_{doc_num:06}_chk_{chunk_num:06}"
                #print(f"Chunk {chunk_num} of document {doc_num}: {split.page_content[:50]}...")  # Print first 50 characters
                split.metadata["doc_id"] = docId
                # also add docid to source
                split.metadata["source"] = f"{docId} - {split.metadata.get('source', '')}"
                split.metadata["chunk_num"] = chunk_num
                split.metadata["doc_num"] = doc_num
                documents.append(split)
                doc_ids.append(f"doc_{doc_num:06}_chk_{chunk_num:06}")

    return documents, doc_ids

def setup(docpath,indexname,local=False):
    # Step 1: Load Docling documents
    documents, doc_ids = loadDocs(docpath)
    print(f"Loaded {len(documents)} chunks from {len(doc_ids)} documents.")
    
    # Step 2: Initialize your custom embedder
    if not local:
        from langchain_community.embeddings import DeepInfraEmbeddings
        embMdl = "BAAI/bge-m3"
        embedder = DeepInfraEmbeddings(model_id=embMdl,deepinfra_api_token=pr.deepInfra["apiKey"])
    else:
        import localEmbeddings as LE
        embedder = LE.LocalEmbeddings("bge-m3-Q4_K_M")

    # Initialize FAISS vector store
    index = faiss.IndexFlatL2(len(embedder.embed_query("hello world")))
    vectorstore = FAISS(
        embedding_function=embedder,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    # Index chunks
    print(f"Indexing {len(documents)} chunks from {len(doc_ids)} documents...")
    idx = vectorstore.add_documents(documents=documents, ids=doc_ids)
    print(f"Indexed {len(documents)} chunks from {len(doc_ids)} documents.",idx)
    # vectorstore = FAISS.from_documents(documents, embedder)
    vectorstore.save_local(indexname)  # Saves to ./faiss_index folder

    if SAVE_DOCS:
        # Save documents to a separate file
        with open(f"{indexname}_docs.json", "w", encoding="utf-8") as f:
            json.dump([{"text":doc.page_content,"meta":doc.metadata} for doc in documents], f, indent=4)
        print(f"Documents saved to {indexname}_docs.json")

    print("FAISS index created and saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Create a FAISS index from Docling documents.")
    parser.add_argument("docpath", type=str, help="Path to the Docling documents.")
    parser.add_argument("indexname", type=str, help="Name of the FAISS index to be created.")
    parser.add_argument("--local", action='store_true', help="Use local embeddings instead of DeepInfra.")

    args = parser.parse_args()

    setup(args.docpath, args.indexname, args.local)

if __name__ == "__main__":
    main()

