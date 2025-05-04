# top2vec needs py y 3.13
import pandas as pd
from top2vec import Top2Vec
import nltk
from nltk.corpus import stopwords

#df = pd.read_csv("karis_de_titles.csv")
df = pd.read_csv("ksk_de_content.csv")
documents = [row.strip() for row in df.content.values]

german_stopwords = set(stopwords.words("german"))
custom_words = {
    "uber",
    "konnen",
    "neue",
    "geht",
    "dabei",
    "de",
    "weitere",
    "sollen",
    "können",
    "derzeit",
    "www",
    "co",
    "com",
    "wen",
}
all_stopwords = german_stopwords.union(custom_words)
model = Top2Vec(
    documents,
    embedding_model="universal-sentence-encoder-multilingual",
    speed="learn",
    workers=8,
    umap_args={
        "n_neighbors": 15,
        "n_components": 10,
        "min_dist": 0.1,
        "metric": "cosine",
    },
    hdbscan_args={
        'min_cluster_size': 5,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'
    },
)
model.save("ksk_content_model")
# load like:
# model = Top2Vec.load("ksk_content_model")

# View topics
topic_words, word_scores, topic_nums = model.get_topics()
for topic_num, words in zip(topic_nums, topic_words):
    print(f"Topic {topic_num}: {words}")


cleaned_topics = []

for words in topic_words:
    filtered = [w for w in words if w.lower() not in all_stopwords]
    cleaned_topics.append(filtered)

for i, words in enumerate(cleaned_topics):
    print(f"Topic {topic_nums[i]}: {words}")


# View example documents per topic
docs_topic_0, _, _ = model.search_documents_by_topic(topic_num=0, num_docs=5)
docs_topic_1, _, _ = model.search_documents_by_topic(topic_num=1, num_docs=5)

print("Topic 0 examples:")
for doc in docs_topic_0:
    print("-", doc)

print("\nTopic 1 examples:")
for doc in docs_topic_1:
    print("-", doc)
