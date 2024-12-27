# needs pip install sentence_transformers 

from sentence_transformers import SentenceTransformer

# Choose a small or quantized model, e.g. MiniLM
model_name = 'sentence-transformers/all-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device='cpu')  # Force CPU usage

sentences = [
    "This is a sentence.",
    "This is another sentence."
]
embeddings = model.encode(sentences)  # Returns a NumPy array
print(embeddings.shape)  # e.g. (2, 384)

while True:
    line = input("Enter text")
    if len(line) == 0:
        break
    embeddings = model.encode([line])
    print(embeddings.shape)

