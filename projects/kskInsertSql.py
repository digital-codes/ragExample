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
llmEn = deployUtils.Llm(lang="en")    

# try to get summary file first
try:
    summary = pd.read_json(summaryFile)
except:
    print("Error reading summary file")
    sys.exit()

connString = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
# clean up
sq.DatabaseUtility.delete_all(connString)

db = sq.DatabaseUtility(connString)

# create project
prj = sq.Project(name="ksk", langs="de,en", description="Klimaschutzkonzept Karlsruhe",
                 vectorName="ksk.vec",vectorPath = "./data",
                 indexName="ksk.idx",indexPath = "./data",
                 embedModel=embedder.model, embedSize=embedder.get_size()
                 )

db.insert(prj)

tags = []

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
    tag1 = f"area_{meta['area']}"
    if tag1 not in tags:
        tags.append(tag1)
        tag_ = sq.Tag(name = tag1)
        db.insert(tag_)
    tag2 = f"bundle_{meta['bundle']}"
    if tag2 not in tags:
        tags.append(tag2)
        tag_ = sq.Tag(name = tag2)
        db.insert(tag_)
    dbItem = sq.Item(itemIdx = itemIdx, 
        name = f"{meta['area']}_{meta['bundle']}_{meta['topic']}"
        ) 
    dbItem = db.insert(dbItem)
    db.updateTags(dbItem,[tag1, tag2])
    # insert texts
    ## de
    lang = "de"
    ### content
    txt = sq.Snippet(content=text, lang=lang, itemId = dbItem.id, refIdx = dbItem.itemIdx, type="content")
    txt = db.insert(txt)
    ### title
    txt = sq.Snippet(content=meta["title"], lang=lang, itemId = dbItem.id, refIdx = dbItem.itemIdx, type="title")
    txt = db.insert(txt)
    embedding = embedder.encode(txt.content)
    vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
    binary_data = vector.tobytes()
    dbVector = sq.Vector(snipId = txt.id, value = binary_data)   
    db.insert(dbVector) 
    ### summary
    txt = sq.Snippet(content=item.text, lang=lang, itemId = dbItem.id, refIdx = dbItem.itemIdx, type="summary")
    txt = db.insert(txt)

    ## en
    lang = "en"
    ### content
    txt = sq.Snippet(content=llmEn.translate(text)[0], lang=lang, itemId = dbItem.id, refIdx = dbItem.itemIdx, type="content")
    txt = db.insert(txt)
    ##' title
    txt = sq.Snippet(content=llmEn.translate(meta["title"])[0], lang=lang, itemId = dbItem.id, refIdx = dbItem.itemIdx, type="title")
    txt = db.insert(txt)
    embedding = embedder.encode(txt.content)
    vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
    binary_data = vector.tobytes()
    dbVector = sq.Vector(snipId = txt.id, value = binary_data)   
    db.insert(dbVector) 
    ### summary
    txt = sq.Snippet(content=llmEn.translate(item.text)[0], lang=lang, itemId = dbItem.id, refIdx = dbItem.itemIdx, type="summary")
    txt = db.insert(txt)

    # create chunk vectors
    chunks = preprocessor.chunk(text)
    for i, chunk in enumerate(chunks):
        # create item
        print(f"Processing chunk {i}")
        preview, _ = llm.preview(chunk)
        dbChunk = sq.Chunk(itemId = dbItem.id, chunkIdx = chunkIdx)
        # , text = chunk, preview = preview)
        dbChunk = db.insert(dbChunk)

        ### content
        # de
        txt = sq.Snippet(content=chunk, lang="de", itemId = dbItem.id, chunkId = dbChunk.id, refIdx = dbChunk.chunkIdx, type="content")
        txt = db.insert(txt)
        embedding = embedder.encode(txt.content)
        vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
        binary_data = vector.tobytes()
        dbVector = sq.Vector(snipId = txt.id, value = binary_data)   
        db.insert(dbVector) 
        # en
        txt = sq.Snippet(content=llmEn.translate(chunk)[0], lang="en", itemId = dbItem.id, chunkId = dbChunk.id, refIdx = dbChunk.chunkIdx, type="content")
        txt = db.insert(txt)
        embedding = embedder.encode(txt.content)
        vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
        binary_data = vector.tobytes()
        dbVector = sq.Vector(snipId = txt.id, value = binary_data)   
        db.insert(dbVector) 

        ### summary/preview
        # de
        txt = sq.Snippet(content=llm.preview(chunk)[0], lang="de", itemId = dbItem.id, chunkId = dbChunk.id, refIdx = dbChunk.chunkIdx, type="summary")
        txt = db.insert(txt)
        # en
        txt = sq.Snippet(content=llmEn.translate(txt.content)[0], lang="en", itemId = dbItem.id, chunkId = dbChunk.id, refIdx = dbChunk.chunkIdx, type="summary")
        txt = db.insert(txt)

        chunkIdx += 1 
                
    itemIdx += 1 

