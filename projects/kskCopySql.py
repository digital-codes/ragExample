import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
from ragInstrumentation import measure_execution_time

import ragSqlUtils as sq
import private_remote as pr


src_cs = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'

dstFile = "ksk.db"
dst_cs = f"sqlite:///{dstFile}"
try:
    os.remove(dstFile)
    print(f"Removed {dstFile}")
except FileNotFoundError:
    pass
    
src = sq.DatabaseUtility(src_cs)
dst = sq.DatabaseUtility(dst_cs)

# copy project
prj = src.search(sq.Project)
dst.insert(prj[0]) 
print("Copied project:",prj[0].id)

#verify
prj = dst.search(sq.Project)
print("Target:",prj[0])


# copy tags
tags = src.search(sq.Tag)
for t in tags:
    t.id = None
    dst.insert(t)
print(f"{len(tags)} tags copied")

# copy items
items = src.search(sq.Item)
for i in items:
    i.id = None
    dst.insert(i)
print(f"{len(items)} items copied")

# copy chunks
chunks = src.search(sq.Chunk)
for c in chunks:
    dst.insert(c)
print(f"{len(chunks)} chunks copied")

# copy snippets
snippets = src.search(sq.Snippet)
for s in snippets:
    dst.insert(s)
print(f"{len(snippets)} snippets copied")
    
