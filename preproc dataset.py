import os

DATASET_DIR = r"D:\Tugas Kuliah\Semester 8\Tugas Akhir\presensi_arcface\uji_coba\foto_asli"
ENROLLMENT_RATIO = 0.8
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

def hitung_statistik_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Folder '{DATASET_DIR}' tidak ditemukan.")
        return

    total_siswa = 0
    total_foto = 0
    total_enrollment = 0
    total_testing = 0

    print("=" * 60)
    print(f"{'NAMA SISWA':<25} | {'FOTO':<5} | {'ENROLL':<7} | {'TEST':<5}")
    print("-" * 60)

    # Mengurutkan folder siswa
    daftar_siswa = sorted([d for d in os.listdir(DATASET_DIR) 
                           if os.path.isdir(os.path.join(DATASET_DIR, d))])

    for nama_siswa in daftar_siswa:
        folder_siswa = os.path.join(DATASET_DIR, nama_siswa)
        
        # Filter hanya file gambar
        images = [f for f in os.listdir(folder_siswa) 
                  if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        
        jumlah_foto = len(images)

        # Skip jika foto kurang dari 2
        if jumlah_foto < 2:
            print(f"{nama_siswa:<25} | Skip (kurang dari 2 foto)")
            continue

        enrollment = max(1, int(jumlah_foto * ENROLLMENT_RATIO))
        testing = jumlah_foto - enrollment

        total_siswa += 1
        total_foto += jumlah_foto
        total_enrollment += enrollment
        total_testing += testing

        print(f"{nama_siswa:<25} | {jumlah_foto:<5} | {enrollment:<7} | {testing:<5}")

    print("-" * 60)
    print(f"Total Siswa         : {total_siswa}")
    print(f"Total Foto          : {total_foto}")
    print(f"Enrollment          : {total_enrollment}")
    print(f"Testing             : {total_testing}")
    
    if total_siswa > 0:
        print(f"Rata-rata foto/siswa: {total_foto/total_siswa:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    hitung_statistik_dataset()