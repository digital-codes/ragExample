import ragSqlUtils as sq
import private_remote as pr
import os
import numpy as np 

import ragDeployUtils as deployUtils # for embedder size

cs = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
db = sq.DatabaseUtility(cs)

DIM = deployUtils.Embedder.get_size()

projects = [p.name for p in db.search(sq.Project)]

print("Available projects:",projects)

name = input("Select project: ")

sel = db.search(sq.Project,filters = [sq.Project.name == name])
if len(sel) == 0:
    print("Invalid selection")
    exit()

prId = sel[0].id
path = sel[0].vectorPath
base = sel[0].vectorName
print("ID:",prId)

try:
    os.mkdir(path)
except FileExistsError:
    pass

# chunks
# get chunks, ordered by idx
chunks = db.get_chunks(prId)
print("Chunks:",len(chunks))
fn = os.sep.join([path,f"chunk_{base}"])
with open (fn,"rb") as f:
    data = f.read()
itemSize = np.dtype(np.float32).itemsize
size = len(data)//itemSize
print("Chunk vectors read:",fn,len(chunks),itemSize,size)
assert len(data) == len(chunks) * DIM * itemSize
for c in chunks:
    print("Chunk:",c.id,c.chunkIdx)
    chunkId = c.id
    v = db.search(sq.Vector,filters = [sq.Vector.chunkId == chunkId])[0]
    print("Vector:",v.id,c.chunkIdx)
    vdata = np.frombuffer(data[c.chunkIdx*itemSize*DIM:(c.chunkIdx+1)*itemSize*DIM], dtype='float32')
    assert len(vdata) == DIM
    binary_data = vdata.tobytes()
    assert len(binary_data) == DIM * itemSize
    v.value = binary_data
    db.update(sq.Vector,v)

# titles
items = db.get_items(prId)
print("Items:",len(items))
fn = os.sep.join([path,f"title_{base}"])
with open (fn,"rb") as f:
    data = f.read()
itemSize = np.dtype(np.float32).itemsize
size = len(data)//itemSize
print("Title vectors read:",fn,len(items),itemSize,size)
assert len(data) == len(items) * DIM * itemSize
for i in items:
    print("Item:",i.id,i.itemIdx)
    itemId = i.id
    v = db.search(sq.TitleVector,filters = [sq.TitleVector.itemId == itemId])[0]
    print("Vector:",v.id,i.itemIdx)
    vdata = np.frombuffer(data[i.itemIdx*itemSize*DIM:(i.itemIdx+1)*itemSize*DIM], dtype='float32')
    assert len(vdata) == DIM
    binary_data = vdata.tobytes()
    assert len(binary_data) == DIM * itemSize
    v.value = binary_data
    db.update(sq.TitleVector,v)

