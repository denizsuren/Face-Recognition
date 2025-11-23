import uuid
import cv2
import traceback
import numpy as np
import faiss
import psycopg2
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import shutil
from pydantic import BaseModel
from typing import List
from insightface.app import FaceAnalysis
import os

# =============================
# CONFIG
# =============================
FAISS_INDEX_PATH = "face_index.faiss"
DB_CONFIG = {
    "dbname": "fastapi",
    "user": "postgres",
    "password": "h12345jklj",
    "host": "localhost",
    "port": "5432"
}
SIMILARITY_THRESHOLD = 0.55 # Threshold'u 0.55'e çektik
API_URL = "http://127.0.0.1:8000"

# InsightFace
app_insight = FaceAnalysis(name='buffalo_l')
app_insight.prepare(ctx_id=-1)  # CPU

# FastAPI
app = FastAPI()


# =============================
# DATABASE
# =============================
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


# =============================
# FAISS - Basit Index Mantığı
# =============================
dim = 512
index = faiss.IndexFlatIP(dim)  # IndexIDMap kullanmıyoruz


def load_faiss_index():
    global index
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"[INFO] Faiss index yüklendi: {index.ntotal} embedding")
        print(f"[INFO] Index tipi: {type(index).__name__}")
    except Exception as e:
        print(f"[INFO] Faiss index bulunamadı: {e}")
        index = faiss.IndexFlatIP(dim)
        print(f"[INFO] Yeni boş index oluşturuldu")


def save_faiss_index():
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[INFO] FAISS index kaydedildi: {index.ntotal} embedding")


def get_next_position():
    """Yeni embedding için bir sonraki pozisyonu al (FAISS index boyutu)"""
    return index.ntotal


# =============================
# MODELS
# =============================
class EmbeddingRequest(BaseModel):
    embedding: List[float]


class AddUserRequest(BaseModel):
    name: str
    passport_id: str
    embedding: List[float]


# =============================
# STARTUP
# =============================
@app.on_event("startup")
def startup_event():
    load_faiss_index()


# =============================
# IDENTIFY
# =============================
@app.post("/identify")
def identify(req: EmbeddingRequest):
    try:
        global index

        # InsightFace zaten normalize edilmiş veriyor
        vector = np.array(req.embedding, dtype='float32').reshape(1, -1)

        if index.ntotal == 0:
            return {"status": "new_user", "similarity": 0.0}

        # Basit index'te arama yap
        distances, indices = index.search(vector, 1)
        similarity = float(distances[0][0])
        faiss_position = int(indices[0][0])

        print(
            f"[DEBUG IDENTIFY] similarity={similarity:.3f}, faiss_position={faiss_position}, threshold={SIMILARITY_THRESHOLD}")

        if similarity >= SIMILARITY_THRESHOLD:
            # FAISS pozisyonu = Database user_id
            user_id = faiss_position

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, name, passport_id FROM passports WHERE user_id = %s;", (user_id,))
            user = cur.fetchone()
            cur.close()
            conn.close()

            if user:
                print(f"[DEBUG] Kullanıcı bulundu: {user}")
                return {
                    "status": "matched",
                    "similarity": similarity,
                    "user": {
                        "user_id": user[0],
                        "name": user[1],
                        "passport_id": user[2]
                    }
                }
            else:
                print(f"[DEBUG] FAISS pozisyon {faiss_position} için DB'de user_id {user_id} bulunamadı!")

        return {"status": "new_user", "similarity": similarity}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Identify hatası: {e}")


# =============================
# ADD USER
# =============================
@app.post("/add_user")
def add_user(req: AddUserRequest):
    global index
    try:
        vector = np.array(req.embedding, dtype='float32').reshape(1, -1)

        conn = get_db_connection()
        cur = conn.cursor()

        # Passport ID kontrolü
        cur.execute("SELECT user_id FROM passports WHERE passport_id = %s;", (req.passport_id,))
        existing = cur.fetchone()

        if existing:
            user_id = existing[0]
            cur.close()
            conn.close()
            return {"status": "exists", "user_id": user_id, "message": "Bu passport_id zaten kayıtlı."}

        # Yeni pozisyon al (FAISS index boyutu = yeni pozisyon)
        new_position = get_next_position()

        # Database'e ekle (user_id = FAISS pozisyonu)
        cur.execute(
            "INSERT INTO passports (user_id, name, passport_id) VALUES (%s, %s, %s);",
            (new_position, req.name, req.passport_id)
        )

        # FAISS'e sequential olarak ekle (pozisyon otomatik artacak)
        index.add(vector)
        save_faiss_index()

        conn.commit()
        cur.close()
        conn.close()

        print(f"[DEBUG] Yeni kullanıcı eklendi: Position/ID={new_position}, Name={req.name}")
        return {"status": "user_added", "user_id": new_position, "message": "Kullanıcı başarıyla eklendi."}

    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AddUser hatası: {ex}")

@app.post("/add_user_from_image")
async def add_user_from_image(
    name: str = Form(...),
    passport_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Geçici dosya kaydet
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Resmi oku
        img = cv2.imread(temp_filename)
        if img is None:
            os.remove(temp_filename)
            raise HTTPException(status_code=400, detail="Geçersiz resim dosyası")

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Yüz bul
        faces = app_insight.get(img_rgb)
        if not faces:
            os.remove(temp_filename)
            raise HTTPException(status_code=400, detail="Resimde yüz bulunamadı")

        # İlk yüzün embedding'ini al
        embedding = faces[0].normed_embedding.astype(np.float32).reshape(1, -1)

        # FAISS pozisyonu
        new_position = get_next_position()

        # DB'ye ekle
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO passports (user_id, name, passport_id) VALUES (%s, %s, %s);",
            (new_position, name, passport_id)
        )
        conn.commit()
        cur.close()
        conn.close()

        # FAISS'e ekle
        index.add(embedding)
        save_faiss_index()

        # Geçici dosyayı sil
        os.remove(temp_filename)

        print(f"[DEBUG] Yeni kullanıcı (resimden) eklendi: {name}, ID={new_position}")
        return {
            "status": "user_added",
            "user_id": new_position,
            "name": name,
            "passport_id": passport_id,
            "message": "Kullanıcı resimden eklendi."
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Resimden ekleme hatası: {e}")



# =============================
# DEBUG ENDPOINTS
# =============================
@app.get("/debug/stats")
def debug_stats():
    """Debug bilgileri"""
    try:
        # FAISS stats
        faiss_count = index.ntotal

        # Database stats
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM passports")
        db_count = cur.fetchone()[0]

        cur.execute("SELECT MIN(user_id), MAX(user_id) FROM passports")
        min_max = cur.fetchone()
        min_id, max_id = min_max if min_max[0] is not None else (None, None)

        # Sequential kontrol
        cur.execute("SELECT user_id FROM passports ORDER BY user_id")
        user_ids = [row[0] for row in cur.fetchall()]
        expected_ids = list(range(len(user_ids))) if user_ids else []
        is_sequential = user_ids == expected_ids

        cur.close()
        conn.close()

        return {
            "faiss_embeddings": faiss_count,
            "database_records": db_count,
            "user_id_range": f"{min_id}-{max_id}" if min_id is not None else "empty",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "sequential_ids": is_sequential,
            "index_type": type(index).__name__,
            "status": "ok" if faiss_count == db_count and is_sequential else "mismatch"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/mapping_check")
def debug_mapping_check():
    """FAISS pozisyon ve DB user_id mapping kontrolü"""
    try:
        if index.ntotal == 0:
            return {"message": "Index boş"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id, name FROM passports ORDER BY user_id LIMIT 10")
        db_records = cur.fetchall()
        cur.close()
        conn.close()

        mapping_results = []
        for i in range(min(10, index.ntotal, len(db_records))):
            db_user_id, db_name = db_records[i]
            expected_position = i

            mapping_results.append({
                "faiss_position": expected_position,
                "db_user_id": db_user_id,
                "db_name": db_name[:30] + "..." if len(db_name) > 30 else db_name,
                "mapping_ok": db_user_id == expected_position
            })

        all_mappings_ok = all(r["mapping_ok"] for r in mapping_results)

        return {
            "total_faiss": index.ntotal,
            "total_db": len(db_records),
            "mapping_check": mapping_results,
            "all_mappings_ok": all_mappings_ok,
            "status": "✓ BAŞARILI" if all_mappings_ok else "✗ HATALI"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/test_search")
def test_search():
    """FAISS arama testi"""
    try:
        if index.ntotal == 0:
            return {"error": "Index boş"}

        # Random bir test vektörü ile ara
        test_vector = np.random.random((1, 512)).astype('float32')
        distances, indices = index.search(test_vector, min(3, index.ntotal))

        results = []
        for i in range(len(indices[0])):
            faiss_pos = int(indices[0][i])
            similarity = float(distances[0][i])

            # DB'den kontrol et
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, name FROM passports WHERE user_id = %s", (faiss_pos,))
            user = cur.fetchone()
            cur.close()
            conn.close()

            results.append({
                "rank": i + 1,
                "faiss_position": faiss_pos,
                "similarity": similarity,
                "user_found": user is not None,
                "user_info": {"user_id": user[0], "name": user[1]} if user else None
            })

        return {"test_results": results}
    except Exception as e:
        return {"error": str(e)}


@app.post("/debug/test_identify")
def debug_identify(req: EmbeddingRequest):
    """Debug için detaylı identify testi"""
    try:
        global index

        vector = np.array(req.embedding, dtype='float32').reshape(1, -1)

        # Embedding istatistikleri
        embedding_norm = np.linalg.norm(vector)
        embedding_min = float(np.min(vector))
        embedding_max = float(np.max(vector))

        if index.ntotal == 0:
            return {
                "error": "Index boş",
                "embedding_stats": {
                    "norm": float(embedding_norm),
                    "min": embedding_min,
                    "max": embedding_max
                }
            }

        # En yakın 3 sonucu al
        distances, indices = index.search(vector, min(3, index.ntotal))

        results = []
        for i in range(len(distances[0])):
            similarity = float(distances[0][i])
            faiss_position = int(indices[0][i])
            user_id = faiss_position  # Pozisyon = user_id

            # Database'den kullanıcı bilgisini al
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, name, passport_id FROM passports WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            cur.close()
            conn.close()

            results.append({
                "rank": i + 1,
                "faiss_id": faiss_position,  # Backward compatibility için
                "faiss_position": faiss_position,
                "db_user_id": user_id,
                "similarity": similarity,
                "above_threshold": similarity >= SIMILARITY_THRESHOLD,
                "user_found_in_db": user is not None,
                "mapping_ok": user[0] == faiss_position if user else False,
                "user_info": {
                    "user_id": user[0] if user else None,
                    "name": user[1] if user else None,
                    "passport_id": user[2] if user else None
                } if user else None
            })

        return {
            "threshold": SIMILARITY_THRESHOLD,
            "index_type": type(index).__name__,
            "embedding_stats": {
                "norm": float(embedding_norm),
                "min": embedding_min,
                "max": embedding_max
            },
            "search_results": results,
            "conclusion": {
                "should_match": results[0]["above_threshold"] and results[0]["user_found_in_db"] and results[0][
                    "mapping_ok"] if results else False,
                "top_similarity": results[0]["similarity"] if results else 0
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# =============================
# MAIN - Run FastAPI Server
# =============================
if __name__ == "__main__":
    import uvicorn

    print("[INFO] FastAPI server başlatılıyor...")
    print("[INFO] Server: http://localhost:8000")
    print("[INFO] Debug stats: http://localhost:8000/debug/stats")
    print("[INFO] Mapping check: http://localhost:8000/debug/mapping_check")
    uvicorn.run(app, host="0.0.0.0", port=8000)