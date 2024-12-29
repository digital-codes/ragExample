import json
import os
import datetime
import sys
import pandas as pd
import numpy as np
import ragTextUtils as textUtils
import ragDeployUtils as deployUtils

import ragSqlUtils as sq
import private_remote as pr

DEBUG = False

basedir = '../../js/klimaDashboard/docs/karlsruhe/ksk_extracted'
dbCollection = "ksk_full_de" # kskSum using summary, ksk using raw content
summaryFile = "../../js/klimaDashboard/tools/ksk_summary.json"


# text stuff
preprocessor = textUtils.PreProcessor()

# get models
embedder = deployUtils.Embedder(provider="local")

llm = deployUtils.Llm()    

# try to get summary file first
try:
    summary = pd.read_json(summaryFile)
except:
    print("Error reading summary file")
    sys.exit()

connString = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
db = sq.DatabaseUtility(connString)

# create project
prj = sq.Project(name="ksk", lang="de", description="Klimaschutzkonzept Karlsruhe",vectorName="ksk-de.vec",vectorPath = "./data")
db.insert(prj)

itemIdx = 0
chunkIdx = 0
for item in summary.itertuples(index=False):
    meta = json.loads(item.meta)
    with open(os.path.join(basedir,item.filename.replace(".json",".txt")), 'r') as f:
        print(f"Processing {item.filename}")
        text = f.read()
    # text = item.text
    filename = item.filename
    
    # create item
    dbItem = sq.Item(itemIdx = itemIdx, projectId = prj.id, 
        name = f"{meta['area']}_{meta['bundle']}_{meta['topic']}",
        code = (ord(meta['area']) << 16 ) | (int(meta['bundle']) << 8) | (int(meta['topic'])),
        title = meta['title'],
        fulltext = text
        ) 
    db.insert(dbItem) 
    embedding = embedder.encode(meta['title'])
    vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
    binary_data = vector.tobytes()
    dbTitleVector = sq.TitleVector(itemId = dbItem.id, value = binary_data)   
    db.insert(dbTitleVector) 


    itemIdx += 1 

    chunks = preprocessor.chunk(text)
    for i, chunk in enumerate(chunks):
        # create item
        print(f"Processing chunk {i}")
        preview, _ = llm.preview(chunk)
        dbChunk = sq.Chunk(itemId = dbItem.id, chunkNum = i, chunkIdx = chunkIdx, text = chunk, preview = preview)
        db.insert(dbChunk)
        chunkIdx += 1 
        # create vector
        embedding = embedder.encode(chunk)
        vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
        binary_data = vector.tobytes()
        dbVector = sq.Vector(chunkId = dbChunk.id, value = binary_data)   
        db.insert(dbVector) 
                

