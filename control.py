import faiss
import pickle
import numpy as np

# Faiss index'i yükle
index = faiss.read_index("face_index.faiss")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Faiss index toplam {index.ntotal} embedding içeriyor.")

# Örnek bir arama (ilk embedding ile kendisine bak)
# Normalde yeni bir yüz embedding'i buraya gelir
vectors = np.array([index.reconstruct(0)], dtype='float32')  # ilk vektör
distances, indices = index.search(vectors, k=3)  # en yakın 3 vektör

print("\n--- Arama Sonuçları ---")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. Kullanıcı: {metadata['user_ids'][idx]}, Resim: {metadata['img_names'][idx]}, Mesafe: {distances[0][i]}")