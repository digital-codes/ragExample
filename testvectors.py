import pandas as pd
import argparse 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./rag")))

import ragTextUtils as textUtils
import ragDeployUtils as deployUtils
from ragInstrumentation import measure_execution_time, log_query

import ragSqlUtils as sq
import private_remote as pr

import requests
import time
import json
import numpy as np

lang = "de"
collection = "ksk_1024"
embProvider = "localllama"
llmProvider = "deepinfra"
llmModel = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
dbProvider = "localsearch"
dbSqlite = "projects/ksk.db"

connection_string = f'sqlite:///{dbSqlite}'
sql = sq.DatabaseUtility(connection_string)

# get project infos
prj = sql.search(sq.Project)[0]
print(prj.description)

dbSearch =  {
            "title": deployUtils.VectorDb(provider=dbProvider,collection=f'{collection}_title'),
            #"summary": deployUtils.VectorDb(provider=config["dbProvider"],collection=f'{config["dbCollection"]}_summary'),
            #"chunk": deployUtils.VectorDb(provider=dbProvider,collection=f'{collection}_chunk')
        }
dbClient = dbSearch["title"]
# deployUtils.VectorDb(provider=dbProvider,collection=collection)

# text stuff
preprocessor = textUtils.PreProcessor(lang)
# models
embedder = deployUtils.Embedder(provider=embProvider)

# llm
llm = deployUtils.Llm(lang=lang,provider=llmProvider,model=llmModel)


try:
    db = dbClient.describeCollection()
    print(db)
    if db["code"] != 0:
        print(f"Error on {db}: {db['code']}")
        raise ValueError
    print("Collection OK")
except Exception as e:
    print("Collection failed",e)
    raise ValueError

print("sql",connection_string, sql)
titleItems = sql.search(sq.Item)
print("Titleitems:",len(titleItems))

for i,t in enumerate(titleItems):
    print(i,t.itemIdx,t.id,t.name)

itemIds = [t.id for t in titleItems]


tp = "title"

titles = sql.search(sq.Snippet, filters=[sq.Snippet.lang == lang,
                                          sq.Snippet.type == tp,
                                          sq.Snippet.chunkId == None])
dim = prj.embedSize
size = len(titles)
tv = np.ndarray((size,dim),dtype=np.float32)
queries = []
for t in titles:
    print(t.itemId, t.refIdx)
    query = t.content
    embedding = embedder.encode(query)
    searchVector = embedding["data"][0]["embedding"]
    queries.append({"text":query,"vector":searchVector})
    tv[t.refIdx] = np.array(searchVector).astype(np.float32)
    sr = dbSearch["title"].searchItem(searchVector)
    print(sr)

with open("titles.json","w") as f:
    json.dump(queries,f)

tp = "title"
path = prj.vectorPath
base = prj.vectorName.split(".")[0]
ext = prj.vectorName.split(".")[1]
vector_file = os.sep.join([path,f"{base}_{dim:04}_{tp}_{lang}.{ext}"])
with open(vector_file, 'wb') as f:
    tv.tofile(f)

sys.exit()
for i in range (len(tv)):
    for j in range (len(tv)):
        if i == j:
            continue
        print(i,j,np.dot(tv[i],tv[j]))
        

              

