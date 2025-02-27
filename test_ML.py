from joblib import load
from train_ML import extract_features
import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from display_methods import display_prediction_groundtruth
from sklearn.model_selection import train_test_split
import json
import random

def load_model(filename):
  return load(filename)

def test_sample(sample_file_path, index):
    with h5py.File(sample_file_path[index], 'r') as file:
      image = file['image'][()].transpose(2, 0, 1)
      mask = file['mask'][()].transpose(2, 0, 1)

    return image, mask

def main():
  folder_path = 'Files/Test_Samples'
  test_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

  with open('/content/BraTS_Thesis/config_files/ML_config.json', 'r') as f:
    train_config = json.load(f)

  index = random.randint(0, 149)
  image, mask = test_sample(test_files, index=index)

  nec_mask = mask[0, :, :]
  ed_mask = mask[1, :, :]
  et_mask = mask[2, :, :]

  multi_class_mask = np.zeros_like(nec_mask)
  multi_class_mask[nec_mask > 0] = 1
  multi_class_mask[ed_mask > 0] = 2
  multi_class_mask[et_mask > 0] = 3

  scaler = StandardScaler()
  features = extract_features(image)
  labels = multi_class_mask.flatten()

  X_train, X_test, _, _ = train_test_split(features, labels, test_size=0.2, random_state=42)
  _ = scaler.fit_transform(X_train)
  _ = scaler.transform(X_test)
  features_scaled = scaler.transform(features)
  
  model = train_config['model']
  trained_model = load_model(f'/content/BraTS_Thesis/models_weights/{model}.joblib')

  pred_mask = trained_model.predict(features_scaled)
  pred_mask_image = pred_mask.reshape(multi_class_mask.shape)

  display_prediction_groundtruth(image, pred_mask_image, mask)

if __name__ == "__main__":
    main()