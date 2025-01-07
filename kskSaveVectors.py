import os
import numpy as np
import ragDeployUtils as deployUtils

import ragSqlUtils as sq
import private_remote as pr

DEBUG = False


connString = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
db = sq.DatabaseUtility(connString)
# read project data
prj = db.search(sq.Project)[0]
print(f"Project: {prj.name}")

# get models
embedder = deployUtils.Embedder(provider="localllama")
embedderModel = embedder.model
embedderSize = embedder.get_size()

prj.embedModel = embedderModel
prj.embedSize = embedderSize

db.update(sq.Project,prj)
prj = db.search(sq.Project)[0]
path = prj.vectorPath
base = prj.vectorName.split(".")[0]
ext = prj.vectorName.split(".")[1]
vecMdl = prj.embedModel
vecSize = prj.embedSize

print(f"Embedder: {vecMdl} - {vecSize}")    

for lang in prj.langs.split(","):
    #  do not use content as type directly, see below
    for type in ["title","chunk","summary"]:
        collection = f"{base}_{vecSize:04}_{type}_{lang}"
        vector_file = os.sep.join([path,f"{base}_{vecSize:04}_{type}_{lang}.{ext}"])
        print(f"Processing {collection}")
        print(f"Output to {vector_file}")        


        vectors = []
        # we need to process content for chunks and title and summary for items. do not use content as type directly
        if type == "chunk":
            results = db.search(sq.Snippet, filters=[sq.Snippet.lang==lang, sq.Snippet.type=="content", sq.Snippet.chunkId != None],order_by="refIdx")
        else:
            results = db.search(sq.Snippet, filters=[sq.Snippet.lang==lang, sq.Snippet.type==type, sq.Snippet.chunkId == None],order_by="refIdx")
        print(f"Processing {len(results)}, type {type}, lang {lang}")
        for r in results:
            embedding = embedder.encode(r.content)
            vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
            # Compute the L2 norm
            norm = np.linalg.norm(vector)

            # Normalize the vector
            if norm != 0:
                normalized_vec = vector / norm
            else:
                normalized_vec = vector  # Handle the zero vector case
                
            vectors.append(normalized_vec.astype(np.float32))

        with open(vector_file, 'wb') as f:
            for vec in vectors:
                f.write(vec.tobytes())
                

