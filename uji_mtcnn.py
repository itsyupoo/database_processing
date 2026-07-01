import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

# ==========================================================
# KONFIGURASI
# ==========================================================
DATASET_DIR = r"D:\Tugas Kuliah\Semester 8\Tugas Akhir\presensi_arcface\uji_coba\foto_asli"
OUTPUT_DIR = r"D:\Tugas Kuliah\Semester 8\Tugas Akhir\presensi_arcface\uji_coba\foto_mtcnn"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ==========================================================
# INISIALISASI
# ==========================================================
detector = MTCNN()
total_foto = 0
berhasil_deteksi = 0
gagal_deteksi = 0

# ==========================================================
# BUAT FOLDER OUTPUT
# ==========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# PROSES
# ==========================================================
print("=" * 60)
print("DETEKSI WAJAH MTCNN")
print("=" * 60)

daftar_siswa = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

for nama_siswa in tqdm(daftar_siswa, desc="Memproses Siswa"):
    folder_input = os.path.join(DATASET_DIR, nama_siswa)
    folder_output = os.path.join(OUTPUT_DIR, nama_siswa)

    os.makedirs(folder_output, exist_ok=True)

    images = sorted([f for f in os.listdir(folder_input) if f.lower().endswith(SUPPORTED_EXTENSIONS)])

    for img_name in images:
        total_foto += 1
        img_path = os.path.join(folder_input, img_name)
        img = cv2.imread(img_path)

        if img is None:
            gagal_deteksi += 1
            continue

        h, w = img.shape[:2]

        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale))
            )

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            if len(faces) == 0:
                gagal_deteksi += 1
                continue

            # Ambil wajah terbesar
            wajah_terbesar = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            x, y, w, h = wajah_terbesar["box"]

            x, y = max(0, x), max(0, y)
            face_crop = img[y:y+h, x:x+w]

            if face_crop.size == 0:
                gagal_deteksi += 1
                continue

            output_path = os.path.join(folder_output, img_name)
            cv2.imwrite(output_path, face_crop)
            berhasil_deteksi += 1

        except Exception:
            gagal_deteksi += 1

# ==========================================================
# HASIL
# ==========================================================
detection_rate = (berhasil_deteksi / total_foto * 100) if total_foto > 0 else 0

print("\n" + "=" * 60)
print("HASIL DETECTION RATE MTCNN")
print("=" * 60)
print(f"Total Foto : {total_foto}")
print(f"Berhasil Deteksi : {berhasil_deteksi}")
print(f"Gagal Deteksi : {gagal_deteksi}")
print(f"Detection Rate : {detection_rate:.2f}%")
print("=" * 60)
print("\nFolder hasil:")
print(OUTPUT_DIR)
