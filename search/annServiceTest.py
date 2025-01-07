import json
import requests

query = {
    "limit": 7,
    "vectors": [[0.1] * 384],
    "collection":1
}

r = requests.get("http://localhost:9001")
if r.status_code == 200:
    data = r.json()
    print(data)
else:
    print(r.status_code,r.text)

r = requests.post("http://localhost:9001",json=query)
if r.status_code == 200:
    data = r.json()
    print(data)
else:
    print(r.status_code,r.text)
