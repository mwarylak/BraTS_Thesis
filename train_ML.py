import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from support_methods import sample_h5_file
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import local_binary_pattern
from tempfile import TemporaryFile
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.multiclass import OneVsOneClassifier
import json

def extract_features(image):
    features = []
    for channel in range(image.shape[0]):
        channel_image = image[channel, :, :]
        lbp = local_binary_pattern(channel_image, P=8, R=1.0, method='uniform')
        features.append(channel_image.flatten())  # Intensywność pikseli
        features.append(lbp.flatten())  # LBP jako cechy tekstury
    return np.array(features).T  # wektor cech dla każdego piksela

def train_model(model, images, directory, batch_size, norm_type = ''):
    # Przygotowanie zmiennych tymczasowych do zapisywania cech i etykiet
    print(f'Model: {model}\nNormalization: {norm_type}')
    feature_file = TemporaryFile()
    label_file = TemporaryFile()

    # StandardScaler inicjalizowany przed pętlą
    scaler = StandardScaler()

    files = images
    batch_size = batch_size

    for batch_start in range(0, len(files), batch_size):
        print(f'Batch: {batch_start}')
        batch_files = files[batch_start:batch_start + batch_size]
        batch_features = []
        batch_labels = []

        for file_name in batch_files:
            sample_file_path = os.path.join(directory, file_name)
            with h5py.File(sample_file_path, 'r') as file:
                image = file['image'][()].transpose(2, 0, 1)
                if 'mask' not in file:
                    print(f"No mask found for {file_name}, skipping.")
                    continue
                mask = file['mask'][()].transpose(2, 0, 1)
                #print(f"Processing file: {file_name}")

                #Normalizacja min - max
                if norm_type == 'min-max':
                  #print("MinMAX")
                  for i in range(image.shape[0]):    # Iterate over channels
                    min_val = np.min(image[i])     # Find the min value in the channel
                    image[i] = image[i] - min_val  # Shift values to ensure min is 0
                    max_val = np.max(image[i]) + 1e-4     # Find max value to scale max to 1 now.
                    image[i] = image[i] / max_val

                #Normalizacja Z-score
                elif norm_type == 'z_score':
                  mean = image.mean()
                  std = image.std()
                  image = (image - mean) / std

                #Normalizacja Percentylowa (Clipping)
                elif norm_type == 'percent':
                  p2, p98 = np.percentile(image, (2, 98))
                  image_clipped = np.clip(image, p2, p98)
                  image = (image_clipped - p2) / (p98 - p2)

                else:
                  pass

                # Przygotowanie danych: maska wieloklasowa
                nec_mask = mask[0, :, :]
                ed_mask = mask[1, :, :]
                et_mask = mask[2, :, :]

                multi_class_mask = np.zeros_like(nec_mask)
                multi_class_mask[nec_mask > 0] = 1
                multi_class_mask[ed_mask > 0] = 2
                multi_class_mask[et_mask > 0] = 3

                # Ekstrakcja cech i etykiet
                features = extract_features(image)
                labels = multi_class_mask.flatten()
                #print(f'Feat: {features.shape}, labels: {labels.shape}, image:{image.shape}, mask:{mask.shape}')
                # Dodanie cech i etykiet do bieżącej partii
                batch_features.append(features)
                batch_labels.append(labels)

        # Łączenie danych z partii
        if batch_features:  # Upewnij się, że batch_features nie jest pusty
            batch_features = np.vstack(batch_features)
            batch_labels = np.hstack(batch_labels)

            # Dopasowanie skalera na bieżącej partii i przeskalowanie
            if batch_start == 0:
                scaler.fit(batch_features)  # Dopasowanie skalera na pierwszej partii
            batch_features_scaled = scaler.transform(batch_features)

            # Zapisanie przeskalowanych cech i etykiet do plików tymczasowych
            feature_file.seek(0, os.SEEK_END)
            np.save(feature_file, batch_features_scaled)

            label_file.seek(0, os.SEEK_END)
            np.save(label_file, batch_labels)

    # Wczytanie wszystkich danych cech i etykiet z plików tymczasowych
    feature_file.seek(0)
    label_file.seek(0)

    try:
        all_features = np.load(feature_file, allow_pickle=True)
        all_labels = np.load(label_file, allow_pickle=True)
        print(f"Wczytano {all_features.shape} cech i {all_labels.shape} etykiet.")
    except Exception as e:
        print("Wystąpił problem podczas wczytywania danych:", e)

    # Dodatkowa kontrola przed podziałem na dane treningowe i testowe
    print(f"Liczba próbek: {len(all_features)}, liczba etykiet: {len(all_labels)}")
    if len(all_features) == 0 or len(all_labels) == 0:
        raise ValueError("Brak danych do podziału. Sprawdź, czy obrazy zawierają maski.")

    print("Unique classes in labels:", np.unique(all_labels))

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

    multimodel = model
    multimodel.fit(X_train, y_train)

    y_pred = multimodel.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return y_pred, y_test, multimodel

def save_model(model, filename='model.joblib'):
  dump(model, filename)



def main():
    
    directory = "/content/BraTS2020_training_data/content/data"

    with open("/content/BraTS_Thesis/Files/h5files.txt", "r") as file:
        files = [line.strip() for line in file]

    with open('/content/BraTS_Thesis/ML_config.json', 'r') as f:
        train_config = json.load(f)

    model_type = train_config['model']
    normalization = train_config['normalization_type']
    model_saving = train_config['model_saving']
    batch = train_config['batch_size']

    if model_type == 'SVM':
        model = OneVsOneClassifier(LinearSVC(random_state=42))
    elif model_type == 'Random_Forest':
       model = RandomForestClassifier(n_estimators=100, random_state=42)

    y_pred, y_test, multimodel = train_model(model, files[:200], directory, batch_size=batch, norm_type=normalization)

    if model_saving:
       save_model(multimodel, f'{model_type}.joblib')


if __name__ == "__main__":
    main()