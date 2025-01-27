import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\
/rag')))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../rag')))
import ragDeployUtils as rag
with open("gemet_labels_de.json") as f:
    lbls = json.load(f)

embedder = rag.Embedder(provider="localllama")

vectors = []
vector_file = "gemet_labels_de"

keys = list(lbls.keys())
for k in keys:
    embedding = embedder.encode(k)
    vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
    # Compute the L2 norm
    norm = np.linalg.norm(vector)

    # Normalize the vector
    if norm != 0:
        normalized_vec = vector / norm
    else:
        normalized_vec = vector  # Handle the zero vector case
        
    vectors.append(normalized_vec.astype(np.float32))

with open(f"{vector_file}.vec", 'wb') as f:
    for vec in vectors:
        f.write(vec.tobytes())
        
with open(f"{vector_file}_keys.json", 'w') as f:
    json.dump(keys,f)
