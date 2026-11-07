import json
import os
import cv2
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

"""
TODO:
- Plot histogram (how many imgs in each cluster) -> DONE
- This works really well! With an epsilon of 0.08, I get pretty much no false positives. 
  However there are coins that have different denominations (25 vs 50 schilling) 
  which look the same otherwise!
"""

# --- Konfiguration ---

# 1. Die JSON-Datei, die Ihre Cluster-Ergebnisse (mit group_id) enthält
JSON_IN = "results/2025_08_19_20_14_09/extraction_results_with_groups.json"

# 2. Das Verzeichnis, in dem die *Originalbilder* liegen (z.B. k1_h1_s1_v.jpg)
IMG_DIR = "images"

# 3. Der Hauptordner für die Ausgabe wird jetzt dynamisch bestimmt
# Er wird im selben Verzeichnis wie JSON_IN erstellt.
RESULT_TIMESTAMP_DIR = os.path.dirname(JSON_IN)
REVIEW_DIR = os.path.join(RESULT_TIMESTAMP_DIR, "cluster_review_output")


# ---------------------

def group_data_by_cluster(data):
    """ Gruppiert die JSON-Daten nach 'group_id' """
    print("Gruppiere Münzen nach 'group_id'...")
    groups = {}
    for item in data:
        group_id = item.get('group_id', -1)  # -1 für Münzen ohne Gruppe

        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(item)
    return groups


def create_cluster_histogram(grouped_data, output_dir):
    """
    Erstellt ein Histogramm der Cluster-Größen.
    Zeigt: Wie viele Cluster haben X Bilder?
    """
    print("Erstelle Histogramm...")

    # Filtere die Noise-Gruppe (-1) heraus, falls vorhanden,
    # da diese oft unverhältnismäßig groß ist und die Statistik verzerrt.
    cluster_sizes = [len(items) for gid, items in grouped_data.items() if gid != -1]

    if not cluster_sizes:
        print("Keine validen Cluster-Daten für das Histogramm.")
        return

    plt.figure(figsize=(10, 6))

    # Histogramm erstellen
    # bins='auto' versucht, die optimale Balkenbreite zu finden
    n, bins, patches = plt.hist(cluster_sizes, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Anzahl Bilder im Cluster (Cluster Size)')
    plt.ylabel('Häufigkeit (Anzahl Cluster)')
    plt.title('Verteilung der Cluster-Größen')

    # Mittelwert-Linie hinzufügen
    mean_size = np.mean(cluster_sizes)
    plt.axvline(mean_size, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(mean_size * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(mean_size))

    # Speichern
    plot_filename = os.path.join(output_dir, "cluster_size_histogram.png")
    plt.savefig(plot_filename)
    print(f"Histogramm gespeichert unter: {plot_filename}")

    # Anzeigen (blockiert das Skript, bis das Fenster geschlossen wird)
    # plt.show()

    return plt


def extract_rois_to_folders(grouped_data, img_dir, review_dir):
    """
    Erstellt Ordner für jedes Cluster (mit > 1 Münze) und speichert die ROIs.
    """

    if os.path.exists(review_dir):
        print(f"Lösche alten Review-Ordner: {review_dir}")
        shutil.rmtree(review_dir)
    os.makedirs(review_dir, exist_ok=True)
    print(f"Neuer Review-Ordner erstellt: {review_dir}")

    print("Extrahiere Münz-ROIs in die jeweiligen Gruppenordner (nur Gruppen > 1 Münze)...")

    groups_created = 0

    for group_id, items in tqdm(grouped_data.items(), desc="Verarbeite Cluster"):

        # --- ANFORDERUNG 2: Überspringe Cluster mit 1 oder 0 Münzen ---
        if len(items) <= 1:
            continue

        groups_created += 1

        # Erstelle einen Ordner für diese group_id
        group_folder_name = f"group_{group_id}"
        group_path = os.path.join(review_dir, group_folder_name)
        os.makedirs(group_path, exist_ok=True)

        # Gehe jede Münze in dieser Gruppe durch
        for coin_item in items:
            try:
                # Pfad zum *vollen* Quellbild
                img_path = os.path.join(img_dir, coin_item['front_image'])
                rect = coin_item['front_rect']

                if not os.path.exists(img_path):
                    print(f"Warnung: Bild nicht gefunden: {img_path}. Überspringe.")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warnung: Bild konnte nicht gelesen werden: {img_path}. Überspringe.")
                    continue

                # Extrahiere das Rechteck (ROI)
                x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
                if w <= 0 or h <= 0:
                    continue

                roi = img[y:y + h, x:x + w]

                if roi.size == 0:
                    continue

                # Erstelle einen eindeutigen Dateinamen für die Münze
                output_filename = f"{coin_item['pair_key']}_{coin_item['coin_number']}.jpg"
                output_path = os.path.join(group_path, output_filename)

                cv2.imwrite(output_path, roi)

            except Exception as e:
                print(f"Fehler bei Verarbeitung von Münze {coin_item.get('pair_key', 'N/A')}: {e}")

    if groups_created == 0:
        print("\nKeine Cluster mit mehr als einer Münze gefunden.")
    else:
        print(f"\n{groups_created} Cluster-Ordner zur Überprüfung erstellt.")


def main():
    # 1. JSON-Daten laden
    try:
        with open(JSON_IN, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Fehler: Eingabe-JSON-Datei nicht gefunden unter {JSON_IN}")
        return
    except json.JSONDecodeError:
        print(f"Fehler: JSON konnte nicht dekodiert werden aus {JSON_IN}")
        return

    if not data:
        print("Eingabe-JSON ist leer. Beende.")
        return

    # 2. Daten nach 'group_id' gruppieren
    grouped_data = group_data_by_cluster(data)

    # --- NEU: Histogramm erstellen ---
    # Dies wird vor der Extraktion aufgerufen, um sofort Einsicht in die Daten zu geben.
    # Gespeichert wird es im übergeordneten Ergebnisordner.
    plt = create_cluster_histogram(grouped_data, RESULT_TIMESTAMP_DIR)

    # 3. ROIs extrahieren und in Ordner speichern (REVIEW_DIR wird jetzt dynamisch übergeben)
    extract_rois_to_folders(grouped_data, IMG_DIR, REVIEW_DIR)

    print("\nVerarbeitung abgeschlossen.")
    print(f"Überprüfen Sie die Ergebnisse im Ordner: '{REVIEW_DIR}'")

    # Zeigt eine Zusammenfassung der größten Cluster (die > 1 haben)
    print("\nZusammenfassung der Cluster-Größen (Nur Gruppen > 1 Münze, Top 10):")

    # Filtern der Gruppen für die Zusammenfassung
    filtered_groups = {gid: itms for gid, itms in grouped_data.items() if len(itms) > 1}

    if not filtered_groups:
        print("  Keine Cluster mit mehr als einer Münze gefunden.")
        return

    # Sortiere die gefilterten Gruppen
    sorted_groups = sorted(filtered_groups.items(), key=lambda item: len(item[1]), reverse=True)

    for i, (group_id, items) in enumerate(sorted_groups[:10]):
        print(f"  Ordner 'group_{group_id}': {len(items)} Münzen")

    plt.show()


if __name__ == "__main__":
    main()