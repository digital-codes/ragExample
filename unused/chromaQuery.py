import chromadb
import json
import os
import sys

basedir = 'tika2024'
dbDir = './chroma/ka-rat-db-200'
dbCollection = "tika"

chroma_client = chromadb.PersistentClient(path=dbDir)

try:
    collection = chroma_client.get_collection(name=dbCollection)
except:
    print("Collection not found")
    sys.exit()
    
    
# infos
print("Items:",collection.count()) # returns the number of items in the collection
#print(collection.peek()) # returns a list of the first 10 items in the collection

# run some queries

#result = collection.query(include=["documents","metadatas"],n_results=10,query_texts=["haupausschuss"],where_document={"$contains":"klimaschutz"})  

#print("Query results:",len(result))
#for r in result:
#    print(r)

query = input("Enter your query: ")

results = collection.query(
    include=["distances","documents","metadatas"],
    query_texts=[query],
    n_results=3,
    #where={"metadata_field": "is_equal_to_this"},
    #where_document={"$contains":"umweltausschus"}
)

print(results)

