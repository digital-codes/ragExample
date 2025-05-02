import localEmbeddings as LE
import random
embedder = LE.LocalEmbeddings("bge-m3-Q4_K_M")

print(embedder.embed_documents(["Hello", "world"]))
print(embedder.embed_query("Hello"))

for i in range(10):
    random_number = random.randint(10, 30)
    sent1 = ' '.join(["".join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=20)) for _ in range(random_number)])
    sent2 = ' '.join(["".join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=20)) for _ in range(random_number)])
    sent3 = ' '.join(["".join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=20)) for _ in range(random_number)])
    embedder.embed_documents([sent1,sent2,sent3])
    
    embedder.embed_query(sent1) 
    embedder.embed_query(sent2) 
    embedder.embed_query(sent3) 
    
    