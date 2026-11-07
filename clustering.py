"""
Preprocessing:
- Ähnliche Bilder gruppieren (welche Ansätze gibt es hier?) Evtl. pretrained vision transformer/cnn? --> Bilder vektorisieren/embedden, dann Vektoren clustern + je in einen Ordner machen zur visuellen menschlichen Überprüfung + in JSON Group-ID hinzufügen
- Achtung: Paare nicht mergen, da vorder und Rückseite nicht garantiert richtig gelabelt ist und somit Concat falsches Ergebnis erzeugen könnte, alternativ beide Varianten des Vektors probieren oder nur Vorder oder nur Rückseite nehmen
GUI mit Streamlit zum manuellen Labeln:

- Lese JSON, um Bildausschnitte zu extracten --> Resize auf genormte Größe + Zeige Vorder und Rückseite der Münze
- Slider für Gamma
- Slider für Kontrast?
- Vorwärts, zurück, per String bestimmte Münze aussuchen zum weitermachen
- Logging + prints (wo was hinzugefügt, wo stehen geblieben, ab wo gibt es keine manuellen Einträge mehr
- Mehrere Texteingabefelder: Münzname, Jahr, Land, Wert/Zalh auf Münze z.B. 2 (Euro), Mehrere Links (durch newline getrennt) z.B. Verweis auf Numista Eintrag, Ebay etc., Gefundene Werte (je in Kombination mit Fundort/Seite) z.B. auf Ebay für 65€
- Übersicht wie viel % der Münzen/Gruppen bereits gelabelt sind

Licht/Patina: Embeddings sind robust, aber einheitliches Resize + leichte Normalisierung hilft.
Autosave nach jedem Speichern.

Index der letzten gelabelten in separater state.json festhalten:

Ideen:
Ähnlichkeit zeigen: „Zeige mir 12 ähnliche Münzen“ → FAISS + kNN mit deinem Paar-Embedding.

OCR-Assist: Tesseract/OCR (Jahreszahl, „2“, „EURO“) als Auto-Vorschlag für Felder.

Rotations-Widget: 90°-Schritte + Autosave der gewählten Orientierung.

Mehrbenutzer: user ins Label schreiben (hast du schon) + separate Autosaves je Benutzer.

Konflikt-Merge: Wenn mehrere labeln → kleiner Merge-Step (z. B. „neueres labeled_at gewinnt“).

Reporting: CSV-Export (pair_key, coin_number, group_id, Felder…) + pro Gruppe Fortschrittstabelle.

"""

import argparse
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm  # Progress bar

from coins_common import DEFAULT_RUN

# --- CONFIGURATION ---
# Defaults; override with --run / --eps
JSON_IN = os.path.join(DEFAULT_RUN, "extraction_results.json")
IMG_DIR = "images"
JSON_OUT = os.path.join(DEFAULT_RUN, "extraction_results_with_groups.json")

# Clustering Hyperparameters
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# Lower (0.05 - 0.15) = Stricter, fewer false positives, likely more clusters.
# Higher (0.20 - 0.40) = Looser, groups patina differences better, risk of mixing different coins.
CLUSTER_EPS = 0.08  # 0.08 --> No false positives except for denomination


def get_feature_extractor():
    """
    Loads a pretrained ResNet50 and removes the classification head.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ResNet50 on {device}...")

    # Load pretrained weights (IMAGENET1K_V2 is generally better)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Replace the final fully connected layer with Identity to get features
    # ResNet50 output before FC is 2048 dimensions
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    # Get the specific transforms required by this model setup
    preprocess = weights.transforms()

    return model, preprocess, device


def extract_features(data, model, preprocess, device):
    """
    Iterates through JSON data, crops images, and calculates embeddings.
    """
    embeddings = []
    valid_indices = []  # To keep track of which JSON items actually worked

    print(f"Extracting features for {len(data)} coins...")

    with torch.no_grad():  # Disable gradient calculation for speed
        for idx, item in enumerate(tqdm(data)):
            try:
                # 1. Construct path
                img_filename = item.get('front_image')
                if not img_filename:
                    continue

                full_path = os.path.join(IMG_DIR, img_filename)

                # 2. Load Image using OpenCV
                if not os.path.exists(full_path):
                    # print(f"Warning: Image not found {full_path}")
                    continue

                img_cv = cv2.imread(full_path)
                if img_cv is None:
                    continue

                # 3. Crop based on rect
                rect = item.get('front_rect')
                x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']

                # Sanity check for crop coordinates
                img_h, img_w = img_cv.shape[:2]
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    continue

                # Crop
                crop = img_cv[y:y + h, x:x + w]
                if crop.size == 0:
                    continue

                # 4. Convert to PIL and Preprocess
                # OpenCV is BGR, PIL/PyTorch expects RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)

                input_tensor = preprocess(pil_img)
                input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

                # 5. Inference
                features = model(input_tensor)

                # Flatten and move to CPU numpy
                embedding = features.cpu().numpy().flatten()

                embeddings.append(embedding)
                valid_indices.append(idx)

            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue

    return np.array(embeddings), valid_indices


def embed_images(images_bgr, model, preprocess, device):
    """L2-normalised embeddings for a list of BGR images (used by numista.py)."""
    batch = [
        preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        for img in images_bgr
    ]
    with torch.no_grad():
        features = model(torch.stack(batch).to(device)).cpu().numpy()
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def main():
    global JSON_IN, JSON_OUT, CLUSTER_EPS
    parser = argparse.ArgumentParser(description="Cluster duplicate coins")
    parser.add_argument("--run", default=DEFAULT_RUN, help="results/<timestamp> dir")
    parser.add_argument("--eps", type=float, default=CLUSTER_EPS)
    args = parser.parse_args()
    JSON_IN = os.path.join(args.run, "extraction_results.json")
    JSON_OUT = os.path.join(args.run, "extraction_results_with_groups.json")
    CLUSTER_EPS = args.eps

    # 1. Load Data
    if not os.path.exists(JSON_IN):
        print(f"Error: Input file {JSON_IN} not found.")
        return

    with open(JSON_IN, 'r') as f:
        data = json.load(f)

    # --- DELETE OLD CLUSTER RESULTS ---
    for item in data:
        if 'group_id' in item:
            item['group_id'] = -1

    # 2. Prepare Model
    model, preprocess, device = get_feature_extractor()

    # 3. Extract Features
    # We only use the 'front' image/rect as requested to avoid misalignment issues
    embeddings, valid_indices = extract_features(data, model, preprocess, device)

    if len(embeddings) == 0:
        print("No valid images found or extracted. Exiting.")
        return

    print(f"Successfully embedded {len(embeddings)} coins.")

    # 4. Cluster with DBSCAN
    # metric='cosine' calculates 1 - cosine_similarity
    print(f"Clustering with DBSCAN (eps={CLUSTER_EPS}, metric='cosine')...")

    db = DBSCAN(eps=CLUSTER_EPS, min_samples=1, metric='cosine', n_jobs=-1)
    labels = db.fit_predict(embeddings)

    # 5. Assign Labels back to Data
    # Initialize all group_ids to -1 (error/unprocessed)
    for item in data:
        item['group_id'] = -1

    # Map valid indices to labels
    for i, valid_idx in enumerate(valid_indices):
        data[valid_idx]['group_id'] = int(labels[i])

    # 6. Statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    print("\n--- Results ---")
    print(f"Total Clusters Found: {n_clusters}")

    # Count size of each cluster
    cluster_counts = {k: 0 for k in unique_labels}
    for label in labels:
        cluster_counts[label] += 1

    print("\nCluster Sizes (First 10):")
    sorted_counts = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
    for label, count in sorted_counts[:10]:
        print(f"  Group ID {label}: {count} coins")

    # 7. Save Output
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"\nSaved clustered data to: {JSON_OUT}")


if __name__ == "__main__":
    main()