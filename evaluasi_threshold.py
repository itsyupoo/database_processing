import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
import csv
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_UJI       = "face_database_uji.pkl"
DATABASE_PATH     = "face_database_enrollment.pkl"
MODEL_NAME        = "ArcFace"
DETECTOR          = "skip"   # Wajah sudah terdeteksi/di-crop sebelumnya (hasil MTCNN)

THRESHOLD_MIN     = 0.20
THRESHOLD_MAX     = 0.50
THRESHOLD_STEP    = 0.01

OUTPUT_CSV_EVAL       = "hasil_evaluasi_threshold.csv"
OUTPUT_CSV_DISTANCE   = "hasil_distance_pairs.csv"
OUTPUT_PNG_HIST       = "histogram_intra_inter.png"
OUTPUT_PNG_FARFRR     = "grafik_far_frr.png"
OUTPUT_PNG_METRICS    = "grafik_metrik_threshold.png"
OUTPUT_PNG_CONFUSION = "confusion_matrix.png"


# ============================================================
# FUNGSI: Cosine Distance
# ============================================================
def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - cos_sim


# ============================================================
# STEP 1: Load Database (Berisi List Embedding per Siswa, Tanpa Centroid)
# ============================================================
def load_database():
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(
            f"Database '{DATABASE_PATH}' tidak ditemukan."
        )
    with open(DATABASE_PATH, "rb") as f:
        database = pickle.load(f)
    print("\n===== JUMLAH EMBEDDING DATABASE =====")
    for nama, emb in database.items():
        print(nama, len(emb))
    total_embedding = sum(len(v) for v in database.values())
    print(
        f"[INFO] Database dimuat. "
        f"Total siswa: {len(database)}, "
        f"Total embedding referensi: {total_embedding}"
    )

    return database


# ============================================================
# STEP 2: Ekstraksi Embedding dari Seluruh Foto Data Uji
# ============================================================
def extract_test_embeddings():
    # Pastikan path ini menunjuk ke "face_database_uji.pkl" hasil ekstraksi Anda
    if not os.path.exists(DATASET_UJI): 
        raise FileNotFoundError(f"File '{DATASET_UJI}' tidak ditemukan.")

    with open(DATASET_UJI, "rb") as f:
        data_raw = pickle.load(f) # Ini adalah dict: {nama: [list_foto]}

    # KONVERSI ke format FLAT LIST agar sesuai dengan kebutuhan fungsi lain
    test_data = []
    for nama_siswa, list_foto in data_raw.items():
        for item in list_foto:
            test_data.append({
                "nama_asli": nama_siswa,
                "file": item["filename"],
                "embedding": item["embedding"]
            })
    
    print(f"[INFO] Data uji dimuat. Total sampel: {len(test_data)}")
    return test_data

# ============================================================
# STEP 3: Cari Distance TERKECIL dari Foto Uji terhadap
#          SEMUA Embedding Individual Milik Satu Siswa
# (Pure ArcFace -> tidak ada centroid, jadi dicari embedding
#  referensi mana yang paling mirip dari koleksi embedding siswa itu)
# ============================================================
def min_distance_to_subject(embedding_uji, list_embedding_subjek):
    # list_embedding_subjek sekarang berisi dictionary, kita harus ambil "embedding"-nya saja
    distances = []
    for item in list_embedding_subjek:
        # Mengambil array embedding dari dictionary
        emb_ref = item["embedding"] 
        # Menghitung jarak
        dist = cosine_distance(embedding_uji, emb_ref)
        distances.append(dist)
    
    return min(distances)


# ============================================================
# STEP 4: Hitung Semua Pasangan Distance (Data Uji vs Setiap Siswa)
# Distance yang dicatat = JARAK TERKECIL antara foto uji
# dengan SELURUH embedding referensi milik siswa tersebut
# ============================================================
def compute_all_distances(test_data, database):
    print("\n[PROSES] Menghitung cosine distance (pure ArcFace, tanpa centroid)...")
    all_pairs = []

    for data in test_data:
        nama_asli = data["nama_asli"]
        embedding_uji = data["embedding"]

        for nama_siswa, list_embedding_referensi in database.items():
            distance = min_distance_to_subject(embedding_uji, list_embedding_referensi)
            jenis = "intra" if nama_asli == nama_siswa else "inter"

            all_pairs.append({
                "file": data["file"],
                "nama_asli": nama_asli,
                "nama_dibandingkan": nama_siswa,
                "distance": distance,
                "jenis": jenis
            })

    print(f"[INFO] Total pasangan dihitung: {len(all_pairs)}")
    return all_pairs


# ============================================================
# STEP 5: Tentukan Prediksi Sistem per Foto (Distance Terkecil Antar Siswa)
# ============================================================
def get_predictions(test_data, database, threshold):
    predictions = []

    for data in test_data:
        nama_asli = data["nama_asli"]
        embedding_uji = data["embedding"]

        distances = {}
        for nama_siswa, list_embedding_referensi in database.items():
            distances[nama_siswa] = min_distance_to_subject(embedding_uji, list_embedding_referensi)

        nama_prediksi = min(distances, key=distances.get)
        distance_terkecil = distances[nama_prediksi]

        if distance_terkecil <= threshold:
            hasil_prediksi = nama_prediksi
        else:
            hasil_prediksi = "Tidak Dikenali"

        predictions.append({
            "file": data["file"],
            "nama_asli": nama_asli,
            "prediksi": hasil_prediksi,
            "distance": distance_terkecil
        })

    return predictions


# ============================================================
# STEP 6: Hitung Confusion Matrix (TP, FP, FN)
# ============================================================
def compute_confusion_matrix(predictions):
    TP = 0
    FP = 0
    FN = 0

    for p in predictions:
        if p["prediksi"] == "Tidak Dikenali":
            FN += 1
        elif p["prediksi"] == p["nama_asli"]:
            TP += 1
        else:
            FP += 1

    return TP, FP, FN


# ============================================================
# STEP 7: Hitung Metrik Evaluasi
# ============================================================
def compute_metrics(TP, FP, FN):
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1


# ============================================================
# STEP 8: Hitung FAR/FRR dari SEMUA pasangan intra & inter
# ============================================================
def compute_far_frr_from_pairs(all_pairs, threshold):
    false_accept = 0
    total_inter = 0
    false_reject = 0
    total_intra = 0

    for pair in all_pairs:
        if pair["jenis"] == "intra":
            total_intra += 1
            if pair["distance"] > threshold:
                false_reject += 1
        else:
            total_inter += 1
            if pair["distance"] <= threshold:
                false_accept += 1

    far = false_accept / total_inter if total_inter > 0 else 0
    frr = false_reject / total_intra if total_intra > 0 else 0
    return far, frr

# ============================================================
# CONFUSION MATRIX BERDASARKAN PAIRS (2x2)
# ============================================================
def compute_confusion_matrix_pairs(all_pairs, threshold):
    TP = FP = FN = TN = 0
    for pair in all_pairs:
        if pair["jenis"] == "intra":
            if pair["distance"] <= threshold:
                TP += 1
            else:
                FN += 1
        else:
            if pair["distance"] <= threshold:
                FP += 1
            else:
                TN += 1
    return TP, FP, FN, TN

# ============================================================
# STEP 9: Threshold Search (Loop Semua Threshold)
# ============================================================
def threshold_search(test_data, database, all_pairs):
    print("\n[PROSES] Threshold search 0.20 - 0.60...")
    results = []

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 0.001, THRESHOLD_STEP)

    for threshold in thresholds:
        threshold = round(threshold, 2)
        predictions = get_predictions(test_data, database, threshold)
        TP, FP, FN = compute_confusion_matrix(predictions)
        accuracy, precision, recall, f1 = compute_metrics(TP, FP, FN)
        far, frr = compute_far_frr_from_pairs(all_pairs, threshold)
        false_accept = [
            p for p in all_pairs
            if p["jenis"]=="inter"
            and p["distance"]<=threshold
        ]

        false_reject = [
            p for p in all_pairs
            if p["jenis"]=="intra"
            and p["distance"]>threshold
        ]

        print(f"\nThreshold {threshold}")
        print(f"False Accept = {len(false_accept)}")
        print(f"False Reject = {len(false_reject)}")

        for p in false_reject[:10]:
            print(
                p["nama_asli"],
                p["file"],
                round(p["distance"],3)
            )

        results.append({
            "threshold": threshold,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "FAR": far,
            "FRR": frr
        })

        print(f"   Threshold {threshold:.2f} -> Acc={accuracy:.3f}, Prec={precision:.3f}, "
              f"Rec={recall:.3f}, F1={f1:.3f}, FAR={far:.3f}, FRR={frr:.3f}")

    return results


# ============================================================
# STEP 10: Simpan CSV Hasil Evaluasi Threshold
# ============================================================
def save_evaluation_csv(results):
    with open(OUTPUT_CSV_EVAL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SELESAI] CSV hasil evaluasi threshold disimpan: '{OUTPUT_CSV_EVAL}'")


# ============================================================
# STEP 11: Simpan CSV Seluruh Pasangan Distance
# ============================================================
def save_distance_pairs_csv(all_pairs):
    with open(OUTPUT_CSV_DISTANCE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_pairs[0].keys())
        writer.writeheader()
        writer.writerows(all_pairs)
    print(f"[SELESAI] CSV seluruh pasangan distance disimpan: '{OUTPUT_CSV_DISTANCE}'")


# ============================================================
# STEP 12: Plot Histogram Intra vs Inter-class
# ============================================================
def plot_histogram_intra_inter(all_pairs):
    intra_distances = [p["distance"] for p in all_pairs if p["jenis"] == "intra"]
    inter_distances = [p["distance"] for p in all_pairs if p["jenis"] == "inter"]

    plt.figure(figsize=(8, 5))
    plt.hist(intra_distances, bins=30, alpha=0.6, label="Intra-class (Orang Sama)", color="blue")
    plt.hist(inter_distances, bins=30, alpha=0.6, label="Inter-class (Orang Berbeda)", color="red")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Jumlah Pasangan")
    plt.title("Distribusi Cosine Distance: Intra-class vs Inter-class (Pure ArcFace)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_HIST, dpi=200)
    plt.close()
    print(f"[SELESAI] Histogram disimpan: '{OUTPUT_PNG_HIST}'")


# ============================================================
# STEP 13: Plot Grafik FAR vs FRR terhadap Threshold
# ============================================================
def plot_far_frr_curve(results):
    thresholds = [r["threshold"] for r in results]
    far_values = [r["FAR"] for r in results]
    frr_values = [r["FRR"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far_values, marker="o", label="FAR (False Acceptance Rate)", color="red")
    plt.plot(thresholds, frr_values, marker="o", label="FRR (False Rejection Rate)", color="blue")
    plt.xlabel("Threshold (Cosine Distance)")
    plt.ylabel("Rate")
    plt.title("Grafik FAR vs FRR terhadap Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_FARFRR, dpi=200)
    plt.close()
    print(f"[SELESAI] Grafik FAR-FRR disimpan: '{OUTPUT_PNG_FARFRR}'")

# ============================================================
# Plot Confusion Matrix 2x2
# ============================================================
def plot_confusion_matrix_pairs(TP, FP, FN, TN):

    # ======================================================
    # Confusion Matrix (Count)
    # ======================================================
    cm_count = np.array([
        [TP, FN],
        [FP, TN]
    ], dtype=np.float64)

    # ======================================================
    # Row Normalization
    # Genuine dan Impostor dinormalisasi masing-masing
    # ======================================================
    cm_norm = cm_count.copy()

    cm_norm[0] = cm_norm[0] / cm_norm[0].sum()
    cm_norm[1] = cm_norm[1] / cm_norm[1].sum()

    total_genuine = TP + FN
    total_impostor = FP + TN

    fig, ax = plt.subplots(figsize=(8, 6.8))

    # Heatmap memakai data normalisasi
    im = ax.imshow(
        cm_norm,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Normalized Percentage",
        fontsize=12
    )

    # ======================================================
    # Axis
    # ======================================================
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "Predicted\nGenuine",
            "Predicted\nImpostor"
        ],
        fontsize=12,
        fontweight="bold"
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        [
            f"True\nGenuine\n(n={total_genuine})",
            f"True\nImpostor\n(n={total_impostor})"
        ],
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel(
        "True Label",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=16,
        fontweight="bold"
    )

    # ======================================================
    # Grid putih
    # ======================================================
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)

    ax.grid(
        which="minor",
        color="white",
        linewidth=3
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    labels = [
        ["TP", "FN"],
        ["FP", "TN"]
    ]

    # ======================================================
    # Isi setiap kotak
    # ======================================================
    for i in range(2):
        for j in range(2):

            count = int(cm_count[i, j])
            percent = cm_norm[i, j] * 100

            color = "white" if cm_norm[i, j] > 0.5 else "black"

            ax.text(
                j,
                i,
                f"{labels[i][j]}\n"
                f"{count:,}\n"
                f"({percent:.2f}%)",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PNG_CONFUSION,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SELESAI] Confusion Matrix disimpan: {OUTPUT_PNG_CONFUSION}")
    
# ============================================================
# STEP 14: Plot Grafik Metrik vs Threshold
# ============================================================
def plot_metrics_curve(results):
    thresholds = [r["threshold"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1_score"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracy, marker="o", label="Accuracy")
    plt.plot(thresholds, precision, marker="o", label="Precision")
    plt.plot(thresholds, recall, marker="o", label="Recall")
    plt.plot(thresholds, f1, marker="o", label="F1-Score")
    plt.xlabel("Threshold (Cosine Distance)")
    plt.ylabel("Nilai Metrik")
    plt.title("Perbandingan Metrik Evaluasi terhadap Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_METRICS, dpi=200)
    plt.close()
    print(f"[SELESAI] Grafik metrik disimpan: '{OUTPUT_PNG_METRICS}'")


# ============================================================
# STEP 15: Tentukan Threshold Optimal (F1-Score Tertinggi)
# ============================================================
def find_optimal_threshold(results):

    # ===============================
    # Filter berdasarkan performa minimum
    # ===============================
    candidates = [
        r for r in results
        if r["accuracy"]  >= 0.90
        and r["precision"] >= 0.90
        and r["recall"]    >= 0.90
        and r["f1_score"]  >= 0.90
    ]

    # Jika tidak ada yang memenuhi,
    # gunakan seluruh threshold
    if len(candidates) == 0:
        candidates = results

    # ===============================
    # Urutan prioritas:
    # FAR paling kecil
    # FRR paling kecil
    # F1 terbesar
    # Accuracy terbesar
    # ===============================

    best = sorted(
        candidates,
        key=lambda r: (
            r["FAR"],             # semakin kecil semakin baik
            r["FRR"],             # semakin kecil semakin baik
            -r["f1_score"],       # semakin besar semakin baik
            -r["accuracy"]        # semakin besar semakin baik
        )
    )[0]

    print("\n[HASIL] Threshold optimal berdasarkan multi-kriteria:")
    print(f"   Threshold = {best['threshold']}")
    print(f"   Accuracy  = {best['accuracy']:.4f}")
    print(f"   Precision = {best['precision']:.4f}")
    print(f"   Recall    = {best['recall']:.4f}")
    print(f"   F1-Score  = {best['f1_score']:.4f}")
    print(f"   FAR       = {best['FAR']:.4f}")
    print(f"   FRR       = {best['FRR']:.4f}")

    return best


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    database = load_database()
    test_data = extract_test_embeddings()

    if not test_data:
        print("[ERROR] Tidak ada data uji yang berhasil diekstrak embeddingnya.")
        return

    all_pairs = compute_all_distances(test_data, database)
    # ============================================================
    # DEBUG : TOP INTER PALING MIRIP
    # ============================================================
    print("\n========== TOP 30 INTER PALING MIRIP ==========")

    inter_pairs = sorted(
        [p for p in all_pairs if p["jenis"] == "inter"],
        key=lambda x: x["distance"]
    )

    for p in inter_pairs[:30]:
        print(
            f"Asli : {p['nama_asli']} | "
            f"Dibandingkan : {p['nama_dibandingkan']} | "
            f"File : {p['file']} | "
            f"Distance : {p['distance']:.4f}"
        )

    # ============================================================
    # DEBUG : TOP INTRA TERBURUK
    # ============================================================
    print("\n========== TOP 20 INTRA TERBURUK ==========")

    intra_pairs = sorted(
        [p for p in all_pairs if p["jenis"] == "intra"],
        key=lambda x: x["distance"],
        reverse=True
    )

    for p in intra_pairs[:20]:
        print(
            f"{p['nama_asli']} | "
            f"{p['file']} | "
            f"Distance : {p['distance']:.4f}"
        )

    save_distance_pairs_csv(all_pairs)
    plot_histogram_intra_inter(all_pairs)
    intra = [p["distance"] for p in all_pairs if p["jenis"]=="intra"]
    inter = [p["distance"] for p in all_pairs if p["jenis"]=="inter"]

    print("\n===== STATISTIK DISTANCE =====")
    print(f"Jumlah intra : {len(intra)}")
    print(f"Jumlah inter : {len(inter)}")

    print(f"\nINTRA")
    print(f"Min  : {min(intra):.4f}")
    print(f"Max  : {max(intra):.4f}")
    print(f"Mean : {np.mean(intra):.4f}")
    print(f"P95  : {np.percentile(intra,95):.4f}")
    print(f"P99  : {np.percentile(intra,99):.4f}")

    print(f"\nINTER")
    print(f"Min  : {min(inter):.4f}")
    print(f"Max  : {max(inter):.4f}")
    print(f"Mean : {np.mean(inter):.4f}")
    print(f"P5   : {np.percentile(inter,5):.4f}")
    print(f"P10  : {np.percentile(inter,10):.4f}")

    results = threshold_search(test_data, database, all_pairs)
    save_evaluation_csv(results)
    plot_far_frr_curve(results)
    plot_metrics_curve(results)

    best = find_optimal_threshold(results)

    TP, FP, FN, TN = compute_confusion_matrix_pairs(
        all_pairs,
        best["threshold"]
    )

    print("\n===== CONFUSION MATRIX =====")
    print(f"TP = {TP}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"TN = {TN}")

    plot_confusion_matrix_pairs(
        TP,
        FP,
        FN,
        TN
    )

    print("\n[SEMUA PROSES SELESAI]")
    print("File yang dihasilkan:")
    print(f"  - {OUTPUT_CSV_EVAL}")
    print(f"  - {OUTPUT_CSV_DISTANCE}")
    print(f"  - {OUTPUT_PNG_HIST}")
    print(f"  - {OUTPUT_PNG_FARFRR}")
    print(f"  - {OUTPUT_PNG_METRICS}")
    print(f"  - {OUTPUT_PNG_CONFUSION}")

if __name__ == "__main__":
    main()