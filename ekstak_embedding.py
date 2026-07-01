import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
from deepface import DeepFace

# ==========================================================
# KONFIGURASI
# ==========================================================
DATASET_DIR = r"D:\Tugas Kuliah\Semester 8\Tugas Akhir\dataset"

OUTPUT_DATABASE = "file_database_realtime.pkl"

MODEL_NAME = "ArcFace"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ==========================================================
# INISIALISASI
# ==========================================================
detector = MTCNN()

database = {}

total_siswa = 0
total_foto = 0
berhasil = 0
gagal = 0

# ==========================================================
# FUNGSI EKSTRAKSI EMBEDDING
# ==========================================================
def ekstrak_embedding(face_img):
    """
    face_img berupa numpy array hasil crop wajah dari MTCNN
    """

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    result = DeepFace.represent(
        img_path=face_rgb,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )

    embedding = np.array(result[0]["embedding"], dtype=np.float32)

    return embedding


# ==========================================================
# PROSES PEMBUATAN DATABASE
# ==========================================================
print("=" * 60)
print("MEMBANGUN DATABASE WAJAH REAL-TIME")
print("=" * 60)

daftar_siswa = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

for nama_siswa in tqdm(daftar_siswa, desc="Memproses Siswa"):

    total_siswa += 1

    folder_siswa = os.path.join(DATASET_DIR, nama_siswa)

    database[nama_siswa] = []

    images = sorted([
        f for f in os.listdir(folder_siswa)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    print(f"\n>> {nama_siswa}")

    for img_name in images:

        total_foto += 1

        img_path = os.path.join(folder_siswa, img_name)

        try:

            img = cv2.imread(img_path)

            if img is None:
                gagal += 1
                print(f"   ✗ {img_name} (gambar rusak)")
                continue

            # Resize jika terlalu besar
            h, w = img.shape[:2]

            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                img = cv2.resize(
                    img,
                    (
                        int(w * scale),
                        int(h * scale)
                    )
                )

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = detector.detect_faces(rgb)

            if len(faces) == 0:
                gagal += 1
                print(f"   ✗ {img_name} (wajah tidak terdeteksi)")
                continue

            # Ambil wajah terbesar
            face = max(
                faces,
                key=lambda f: f["box"][2] * f["box"][3]
            )

            x, y, w, h = face["box"]

            x = max(0, x)
            y = max(0, y)

            face_crop = img[y:y+h, x:x+w]

            if face_crop.size == 0:
                gagal += 1
                print(f"   ✗ {img_name} (crop kosong)")
                continue

            embedding = ekstrak_embedding(face_crop)

            database[nama_siswa].append({

                "filename": img_name,

                "embedding": embedding

            })

            berhasil += 1

            print(f"   ✓ {img_name}")

        except Exception as e:

            gagal += 1

            print(f"   ✗ {img_name}")

            print("     ", e)


# ==========================================================
# SIMPAN DATABASE
# ==========================================================
with open(OUTPUT_DATABASE, "wb") as f:

    pickle.dump(database, f)


# ==========================================================
# HASIL
# ==========================================================
print("\n" + "=" * 60)
print("DATABASE BERHASIL DIBUAT")
print("=" * 60)

print(f"Total Siswa      : {total_siswa}")
print(f"Total Foto       : {total_foto}")
print(f"Berhasil         : {berhasil}")
print(f"Gagal            : {gagal}")

print("\nDatabase disimpan menjadi:")

print(OUTPUT_DATABASE)

print("=" * 60)