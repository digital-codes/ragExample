import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
from ragInstrumentation import measure_execution_time

import ragSqlUtils as sq
import private_remote as pr


src_cs = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'

dstFile = "ksk.db"
dst_cs = f"sqlite:///{dstFile}"
sq.DatabaseUtility.delete_all(dst_cs)
    
src = sq.DatabaseUtility(src_cs)
dst = sq.DatabaseUtility(dst_cs)

# copy project
prjs = src.search(sq.Project)
prj = prjs[0]
prj_dict = {key: value for key, value in prj.__dict__.items() if not key.startswith("_") and key != 'id'}
prj = sq.Project(**prj_dict)

dst.insert(prj) 
print("Copied project:",prj.id)
dst.engine.dispose()

dst = sq.DatabaseUtility(dst_cs)

#verify
prj = dst.search(sq.Project)
print("Target project len:",len(prj))


# copy tags
tags = src.search(sq.Tag)
for t in tags:
    _dict = {key: value for key, value in t.__dict__.items() if not key.startswith("_")}
    t = sq.Tag(**_dict)
    dst.insert(t)
print(f"{len(tags)} tags copied")

# copy items
items = src.search(sq.Item)
for i in items:
    _dict = {key: value for key, value in i.__dict__.items() if not key.startswith("_") and key != 'tags'}
    item_tags = src.get_item_tags(i.id)
    ii = sq.Item(**_dict)
    ii.tags = []
    for t in item_tags:
        tag = dst.search(sq.Tag, filters = [sq.Tag.name == t])[0]
        ii.tags.append(tag)
    dst.insert(ii)
print(f"{len(items)} items copied")

# copy chunks
chunks = src.search(sq.Chunk)
for c in chunks:
    _dict = {key: value for key, value in c.__dict__.items() if not key.startswith("_")}
    c = sq.Chunk(**_dict)
    dst.insert(c)
print(f"{len(chunks)} chunks copied")

# copy snippets
snippets = src.search(sq.Snippet)
for s in snippets:
    _dict = {key: value for key, value in s.__dict__.items() if not key.startswith("_")}
    s = sq.Snippet(**_dict)
    dst.insert(s)
print(f"{len(snippets)} snippets copied")
    
