import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

# ==========================================================
# KONFIGURASI
# ==========================================================
DATASET_DIR = r"D:\Tugas Kuliah\Semester 8\Tugas Akhir\presensi_arcface\uji_coba\foto_mtcnn"
MODEL_NAME = "ArcFace"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

OUTPUT_ENROLLMENT = "face_database_enrollment.pkl"
OUTPUT_UJI        = "face_database_uji.pkl"

MIN_FOTO_UJI = 2          # minimal foto disisihkan untuk uji
RANDOM_SEED  = 42         # supaya pembagian acak konsisten setiap kali dijalankan

random.seed(RANDOM_SEED)

# ==========================================================
# FUNGSI: Tentukan Jumlah Foto Uji 80:20
# ==========================================================
ENROLLMENT_RATIO = 0.8

def jumlah_foto_uji(total_foto):
    enrollment = max(1, int(total_foto * ENROLLMENT_RATIO))
    testing = total_foto - enrollment

    # Pastikan minimal ada 1 foto untuk testing
    if testing < 1:
        testing = 1

    return testing


# ==========================================================
# FUNGSI: Ekstrak Embedding dari Satu Foto
# ==========================================================
def ekstrak_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )
    return np.array(result[0]["embedding"], dtype=np.float32)


# ==========================================================
# PROSES UTAMA
# ==========================================================
def main():
    database_enrollment = {}
    database_uji = {}

    total_foto = 0
    berhasil = 0
    gagal = 0

    print("=" * 60)
    print("PEMISAHAN DATA + EKSTRAKSI EMBEDDING ARCFACE")
    print("=" * 60)

    daftar_siswa = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

    for nama_siswa in tqdm(daftar_siswa, desc="Memproses Siswa"):
        folder_siswa = os.path.join(DATASET_DIR, nama_siswa)
        images = sorted([f for f in os.listdir(folder_siswa) if f.lower().endswith(SUPPORTED_EXTENSIONS)])

        if not images:
            continue

        # -----------------------------------------------------
        # STEP 1: Tentukan pembagian enrollment vs uji
        # -----------------------------------------------------
        total = len(images)
        n_uji = jumlah_foto_uji(total)
        n_uji = min(n_uji, total - 1)  # safety, minimal sisakan 1 foto buat enrollment

        images_shuffled = images.copy()
        random.shuffle(images_shuffled)

        foto_uji = images_shuffled[:n_uji]
        foto_enrollment = images_shuffled[n_uji:]

        print(f"\n>> {nama_siswa}: total={total} | enrollment={len(foto_enrollment)} | uji={len(foto_uji)}")

        # -----------------------------------------------------
        # STEP 2: Ekstraksi embedding untuk foto ENROLLMENT
        # -----------------------------------------------------
        database_enrollment[nama_siswa] = []
        for img_name in foto_enrollment:
            total_foto += 1
            img_path = os.path.join(folder_siswa, img_name)
            try:
                embedding = ekstrak_embedding(img_path)
                database_enrollment[nama_siswa].append({
                    "filename": img_name,
                    "embedding": embedding
                })
                berhasil += 1
            except Exception:
                gagal += 1
                print(f"   [GAGAL-ENROLLMENT] {nama_siswa} -> {img_name}")

        # -----------------------------------------------------
        # STEP 3: Ekstraksi embedding untuk foto UJI
        # -----------------------------------------------------
        database_uji[nama_siswa] = []
        for img_name in foto_uji:
            total_foto += 1
            img_path = os.path.join(folder_siswa, img_name)
            try:
                embedding = ekstrak_embedding(img_path)
                database_uji[nama_siswa].append({
                    "filename": img_name,
                    "embedding": embedding
                })
                berhasil += 1
            except Exception:
                gagal += 1
                print(f"   [GAGAL-UJI] {nama_siswa} -> {img_name}")

    # ==========================================================
    # SIMPAN KE DUA FILE TERPISAH
    # ==========================================================
    with open(OUTPUT_ENROLLMENT, "wb") as f:
        pickle.dump(database_enrollment, f)

    with open(OUTPUT_UJI, "wb") as f:
        pickle.dump(database_uji, f)

    # ==========================================================
    # HASIL
    # ==========================================================
    total_enrollment = sum(len(v) for v in database_enrollment.values())
    total_uji = sum(len(v) for v in database_uji.values())

    print("\n" + "=" * 60)
    print("HASIL EKSTRAKSI EMBEDDING")
    print("=" * 60)
    print(f"Total Siswa          : {len(daftar_siswa)}")
    print(f"Total Foto Diproses  : {total_foto}")
    print(f"Berhasil             : {berhasil}")
    print(f"Gagal                : {gagal}")
    print(f"Embedding Enrollment : {total_enrollment}  -> '{OUTPUT_ENROLLMENT}'")
    print(f"Embedding Uji        : {total_uji}  -> '{OUTPUT_UJI}'")
    print("=" * 60)


if __name__ == "__main__":
    main()