from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("BAAI/bge-small-en-v1.5")


sentences = [
    "A government project exceeded its expected construction cost.",
    "The project's actual expenditure was significantly higher than the estimated cost.",
    "The weather is sunny today.",
]


embeddings = model.encode(sentences, normalize_embeddings=True)


def cosine_similarity(a, b):
    return np.dot(a, b)


print("Embedding shape:", embeddings.shape)

print(
    "\nSimilarity between related sentences:",
    cosine_similarity(embeddings[0], embeddings[1]),
)

print(
    "Similarity between unrelated sentences:",
    cosine_similarity(embeddings[0], embeddings[2]),
)
