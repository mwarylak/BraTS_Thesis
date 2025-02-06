import os
import h5py
import numpy as np

def sample_h5_file(dir, files, index, name = ""):
    if name != "":
      sample_file_path = os.path.join(dir, name)
    else:
      sample_file_path = os.path.join(dir, files[index])
    with h5py.File(sample_file_path, 'r') as file:
      image = file['image'][()].transpose(2, 0, 1)
      mask = file['mask'][()].transpose(2, 0, 1)

    print(sample_file_path)

    return image, mask

def remove_files(directory, files):
    h5_files_cleaned = []
    for i in range(len(files)):
        _, mask = sample_h5_file(directory, files, i)
        for channel in range(mask.shape[0]):
            if mask[channel].any():
                h5_files_cleaned.append(files[i])
                break

    with open("h5files.txt", "w") as file:
        for element in h5_files_cleaned:
            file.write(element + "\n")

def count_classes_in_masks(directory, h5_files):
    total_non_zero_counts = [0, 0, 0]

    samples = {0: [], 1: [], 2: []}

    for i in range(len(h5_files)):
        image, mask = sample_h5_file(directory, h5_files, i)

        for channel_index in range(mask.shape[0]): 
            non_zero_count = np.count_nonzero(mask[channel_index])  

            total_non_zero_counts[channel_index] += non_zero_count

            if channel_index in samples:
                samples[channel_index].extend(np.where(mask[channel_index] > 0)[0].tolist())  

    for channel_index in range(len(total_non_zero_counts)):
        print(f'Kanał {channel_index}: {total_non_zero_counts[channel_index]} wartości różne od zera')

