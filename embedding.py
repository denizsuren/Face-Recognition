import os
import cv2
from insightface.app import FaceAnalysis
import pickle
import numpy as np
import faiss
import psycopg2
import random

# PostgreSQL bağlantısı
DB_CONFIG = {
    "dbname": "fastapi",
    "user": "postgres",
    "password": "h12345jklj",
    "host": "localhost",
    "port": "5432"
}

# InsightFace yüz analizi modelini başlat
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # -1 = CPU, 0 = GPU

dataset_path = "images"  # Resimlerin bulunduğu klasör
embedding_list = []

print("Resimlerden embedding çıkarılıyor...")

# Klasördeki tüm dosyaları tara
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)

    # Sadece resim dosyalarını işleme al
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Resim okunamadı: {img_path}")
        continue

    # BGR -> RGB dönüşümü
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = app.get(img_rgb)
    if len(faces) == 0:
        print(f"Yüz bulunamadı: {img_path}")
        continue

    for i, face in enumerate(faces):
        # InsightFace zaten normalize edilmiş embedding veriyor
        embedding = face.normed_embedding.tolist()
        embedding_list.append({
            "embedding": embedding,
            "img_name": f"{img_name}_face{i}",
            "img_path": img_path
        })
        print(f"Embedding çıkarıldı: {img_path} - yüz {i + 1}")

# Embedding'leri kaydet
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embedding_list, f)

print(f"Toplam {len(embedding_list)} embedding kaydedildi.")


def create_faiss_index(embedding_file="embeddings.pkl"):
    """FAISS index oluştur - IndexIDMap olmadan, basit IndexFlatIP"""
    print("FAISS index oluşturuluyor...")

    with open(embedding_file, "rb") as f:
        embedding_list = pickle.load(f)

    if len(embedding_list) == 0:
        raise ValueError("Hiç embedding bulunamadı! Önce embedding çıkar.")

    # Embedding'leri numpy array'e çevir
    vectors = np.array([e['embedding'] for e in embedding_list], dtype='float32')
    img_names = [e['img_name'] for e in embedding_list]

    # Basit IndexFlatIP kullan (IndexIDMap olmadan)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product için normalize edilmiş embedding

    # Sequential olarak ekle (pozisyon = ID)
    index.add(vectors)

    # FAISS index'i kaydet
    faiss.write_index(index, "face_index.faiss")

    # Metadata'yı kaydet - FAISS pozisyonları ile
    metadata = {
        "img_names": img_names,
        "total_count": len(embedding_list)
    }

    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"FAISS index oluşturuldu: {index.ntotal} embedding (IndexFlatIP)")
    print(f"ID sistemi: pozisyon tabanlı (0, 1, 2, ... {index.ntotal - 1})")
    return index, metadata


def init_database():
    """Database tablosunu oluştur"""
    print("Database tablosu kontrol ediliyor...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Tabloyu sil ve yeniden oluştur (temiz başlangıç için)
    cur.execute("DROP TABLE IF EXISTS passports")

    # user_id'yi INTEGER olarak oluştur (SERIAL değil)
    cur.execute("""
        CREATE TABLE passports (
            user_id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            passport_id VARCHAR(50) UNIQUE NOT NULL,
            flight_no VARCHAR(20),
            status VARCHAR(20) DEFAULT 'active',
            score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database tablosu oluşturuldu (user_id = INTEGER)")


def migrate_faiss_metadata_to_db(metadata_file="metadata.pkl"):
    """Metadata'yı database'e aktar - FAISS pozisyonu = Database user_id"""
    print("Metadata database'e aktarılıyor...")

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    img_names = metadata["img_names"]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Sequential pozisyon tabanlı ID'ler (0'dan başlayarak)
    for position, img_name in enumerate(img_names):
        cur.execute("""
            INSERT INTO passports (user_id, name, passport_id, flight_no, status, score)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING;
        """, (
            position,  # FAISS pozisyonu = database user_id
            img_name,  # name alanına img_name
            f"TR-{random.randint(100000, 999999)}",  # passport_id
            f"TK{random.randint(100, 999)}" if random.random() > 0.5 else None,  # flight_no
            'active',  # status
            round(random.uniform(0.8, 1.0), 3)  # score
        ))

    conn.commit()

    # Kontrol
    cur.execute("SELECT COUNT(*) FROM passports")
    count = cur.fetchone()[0]

    cur.execute("SELECT MIN(user_id), MAX(user_id) FROM passports")
    min_id, max_id = cur.fetchone()

    cur.close()
    conn.close()

    print(f"Migration tamamlandı:")
    print(f"  - {count} kayıt eklendi")
    print(f"  - user_id aralığı: {min_id} - {max_id}")
    print(f"  - FAISS pozisyonu = Database user_id eşleşmesi sağlandı")


def debug_faiss_db_mapping():
    """FAISS ve DB arasındaki mapping'i kontrol et"""
    print("\n=== MAPPING KONTROLÜ ===")

    # FAISS index yükle
    try:
        index = faiss.read_index("face_index.faiss")
        print(f"FAISS Total: {index.ntotal}")
    except Exception as e:
        print(f"FAISS index yüklenemedi: {e}")
        return

    # Database'den kayıtları al
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT user_id, name FROM passports ORDER BY user_id")
        db_records = cur.fetchall()
        cur.close()
        conn.close()
        print(f"DB Records: {len(db_records)}")
    except Exception as e:
        print(f"Database hatası: {e}")
        return

    # Metadata yükle
    try:
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        print(f"Metadata: {len(metadata['img_names'])}")
    except Exception as e:
        print(f"Metadata yüklenemedi: {e}")
        return

    # İlk 10 kaydı karşılaştır
    print("\nİlk 10 kayıt mapping kontrolü:")
    for i in range(min(10, len(db_records), len(metadata['img_names']))):
        db_id, db_name = db_records[i]
        metadata_name = metadata['img_names'][i]

        status = "✓" if db_id == i else "✗"
        print(f"{status} Pos {i}: DB_ID={db_id}, DB_Name={db_name[:30]}..., Meta_Name={metadata_name[:30]}...")

        if db_id != i:
            print(f"    ⚠️  ID uyuşmazlığı: Beklenen={i}, Gerçek={db_id}")

    # Özet
    mapping_ok = all(db_records[i][0] == i for i in range(len(db_records)))
    print(f"\nMapping Durumu: {'✓ BAŞARILI' if mapping_ok else '✗ HATALI'}")


def verify_setup():
    """Kurulumu doğrula"""
    print("\nKurulum doğrulanıyor...")

    # FAISS index kontrol
    try:
        index = faiss.read_index("face_index.faiss")
        print(f"✓ FAISS index: {index.ntotal} embedding")
        faiss_count = index.ntotal
    except Exception as e:
        print(f"✗ FAISS index hatası: {e}")
        return False

    # Database kontrol
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM passports")
        db_count = cur.fetchone()[0]

        # ID'lerin sequential olduğunu kontrol et
        cur.execute("SELECT user_id FROM passports ORDER BY user_id")
        ids = [row[0] for row in cur.fetchall()]
        expected_ids = list(range(len(ids)))

        cur.close()
        conn.close()

        print(f"✓ Database: {db_count} kayıt")

        # ID sıralaması kontrol
        if ids == expected_ids:
            print("✓ Database ID'leri sequential (0, 1, 2, ...)")
        else:
            print("✗ Database ID'leri sequential değil!")
            print(f"  Beklenen: {expected_ids[:10]}...")
            print(f"  Gerçek: {ids[:10]}...")
            return False

    except Exception as e:
        print(f"✗ Database hatası: {e}")
        return False

    # Sayı eşleşmesi kontrol
    if faiss_count == db_count:
        print("✓ FAISS ve Database kayıt sayıları eşleşiyor")
        return True
    else:
        print(f"✗ Kayıt sayıları eşleşmiyor! FAISS: {faiss_count}, DB: {db_count}")
        return False


if __name__ == "__main__":
    try:
        print("=== Veri Hazırlama Başlatılıyor ===")

        # 1. FAISS index oluştur
        index, metadata = create_faiss_index()

        # 2. Database'i hazırla
        init_database()

        # 3. Metadata'yı database'e aktar
        migrate_faiss_metadata_to_db()

        # 4. Mapping'i debug et
        debug_faiss_db_mapping()

        # 5. Kurulumu doğrula
        if verify_setup():
            print("\n🎉 Kurulum başarıyla tamamlandı!")
            print("✓ FAISS pozisyonu = Database user_id eşleşmesi sağlandı")
            print("Artık API'yi başlatabilir ve kamera scriptini çalıştırabilirsiniz.")
        else:
            print("\n❌ Kurulumda sorun var, lütfen kontrol edin.")

    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback

        traceback.print_exc()