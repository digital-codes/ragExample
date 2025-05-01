import localEmbeddings as LE

embedder = LE.LocalEmbeddings("bge-m3-Q4_K_M")

print(embedder.embed_documents(["Hello", "world"]))
print(embedder.embed_query("Hello"))
