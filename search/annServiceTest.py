import json
import requests

query = {
    "top_n": 7,
    "vector": [0.1] * 384,
    "vector_index":0
}

r = requests.post("http://localhost:9001",json=query)
if r.status_code == 200:
    data = r.json()
    print(data)
else:
    print(r.status_code,r.text)
