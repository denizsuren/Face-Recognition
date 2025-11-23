import cv2
import requests
from insightface.app import FaceAnalysis
import random
import numpy as np

IDENTIFY_URL = "http://127.0.0.1:8000/identify"
ADD_USER_URL = "http://127.0.0.1:8000/add_user"
DEBUG_STATS_URL = "http://127.0.0.1:8000/debug/stats"
DEBUG_IDENTIFY_URL = "http://127.0.0.1:8000/debug/test_identify"

# InsightFace başlat
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # CPU


def check_api_connection():
    try:
        response = requests.get(DEBUG_STATS_URL, timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"[INFO] API bağlantısı OK - {stats['faiss_embeddings']} kayıt var")
            return True
        else:
            print(f"[ERROR] API yanıt vermedi: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] API'ye bağlanılamadı: {e}")
        return False


def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def capture_best_frame():
    """q'ya basıldığında o anki kareyi yakala"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera açılamadı!")
        return None

    print("[INFO] Kamera açıldı.")
    print("[INFO] Yüzü yakalamak için 'q' tuşuna basın")
    print("[INFO] Çıkmak için 'ESC' tuşuna basın")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Kare okunamadı!")
            break

        # Ekrana göster
        cv2.imshow("Face Recognition Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            captured_frame = frame.copy()
            print("[INFO] Kare yakalandı!")
            break
        elif key == 27:  # ESC
            print("[INFO] İptal edildi.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame



def process_frame_once(frame):
    try:
        print("[INFO] Yüz aranıyor...")
        faces = app.get(frame)
        if not faces:
            print("[ERROR] Karede yüz bulunamadı!")
            return None

        print(f"[INFO] {len(faces)} yüz tespit edildi. İlk yüz işleniyor...")

        # --- Çerçeve çizme ekledik ---
        frame_copy = frame.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("Detected Face", frame_copy)
        cv2.waitKey(1000)  # 1 saniye göster
        cv2.destroyWindow("Detected Face")
        # --- Çerçeve çizme bitti ---

        embedding = faces[0].normed_embedding.tolist()
        debug_response = requests.post(DEBUG_IDENTIFY_URL, json={"embedding": embedding}, timeout=10)
        if debug_response.status_code == 200:
            debug_data = debug_response.json()
            print(f"[DEBUG] Embedding norm: {debug_data.get('embedding_stats', {}).get('norm', 'N/A')}")

        response = requests.post(IDENTIFY_URL, json={"embedding": embedding}, timeout=10)
        if response.status_code != 200:
            print(f"[ERROR] API hatası: {response.status_code} - {response.text}")
            return None

        data = response.json()
        print(f"[DEBUG] API yanıtı: {data}")

        if data.get("status") == "matched":
            user_info = data.get("user", {})
            similarity = data.get("similarity", 0)
            print(f"🎉 Kullanıcı tanındı: {user_info['name']} (Benzerlik: {similarity:.3f})")
            return {"status": "recognized", "user_info": user_info, "similarity": similarity}

        elif data.get("status") == "new_user":
            similarity = data.get("similarity", 0)
            print(f"[INFO] Yeni kullanıcı tespit edildi! (Benzerlik: {similarity:.3f})")
            name = input("Kullanıcı adını girin: ").strip() or f"Person{random.randint(1000, 9999)}"
            passport_id = input("Passport ID girin: ").strip() or f"TR-{random.randint(100000, 999999)}"
            add_response = requests.post(ADD_USER_URL, json={
                "name": name,
                "passport_id": passport_id,
                "embedding": embedding
            }, timeout=10)
            return {"status": "added"} if add_response.status_code == 200 else None

    except Exception as e:
        print(f"[ERROR] Frame işlenirken hata: {e}")
        return None


def main():
    print("🎥 Yüz Tanıma Sistemi")
    if not check_api_connection():
        return

    best_frame = capture_best_frame()
    if best_frame is None:
        print("❌ Kare yakalanamadı.")
        return

    result = process_frame_once(best_frame)
    if result:
        print(f"✅ İşlem başarılı: {result['status']}")
    else:
        print("❌ İşlem başarısız!")


if __name__ == "__main__":
    main()
