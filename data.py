from torch.utils.data import Dataset
import torch
import albumentations as A
import os
import h5py
import numpy as np

def augment_lesion(image, mask):
    augmentation = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ElasticTransform(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
    ], is_check_shapes=False)

    try:
        augmented = augmentation(image=image, mask=mask)
        augmented_images = augmented['image']
        augmented_masks = augmented['mask']
    except Exception as e:
        augmented_images = image
        augmented_masks = mask

    return augmented_images, augmented_masks

def NEC_ET_augmentation(sample, directory, new_directory):

        file_path = os.path.join(directory, sample)
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]

        nec_mask = mask[:, :, 0]
        nec_non_zero_value = np.count_nonzero(nec_mask)
        ed_mask = mask[:, :, 1]
        ed_non_zero_value = np.count_nonzero(ed_mask)
        et_mask = mask[:, :, 2]
        et_non_zero_value = np.count_nonzero(et_mask)

        new_file_path = ''

        if nec_non_zero_value > ed_non_zero_value or et_non_zero_value > ed_non_zero_value:

            combined_mask = nec_mask + et_mask
            non_zero_indices = np.where(combined_mask != 0)

            y_min, y_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            x_min, x_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            bounding_box = (y_min, y_max, x_min, x_max)

            cropped_image = image[y_min:y_max, x_min:x_max, :]
            cropped_mask = mask[y_min:y_max, x_min:x_max, :]

            aug_image, aug_mask = augment_lesion(cropped_image, cropped_mask)

            new_image = image.copy()
            new_mask = mask.copy()
            new_image[y_min:y_max, x_min:x_max, :] = aug_image
            new_mask[y_min:y_max, x_min:x_max, :] = aug_mask

            new_file_name = f"{os.path.basename(file_path).replace('.h5', '')}_lesion_aug.h5"
            new_file_path = os.path.join(new_directory, new_file_name)
            with h5py.File(new_file_path, 'w') as new_file:
                new_file.create_dataset('image', data=new_image)
                new_file.create_dataset('mask', data=new_mask)

        return new_file_path

def augment_image_and_mask(image, mask):
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5)
    ], is_check_shapes=False)

    try:
        augmented = augmentation(image=image, mask=mask)
        augmented_images = augmented['image']
        augmented_masks = augmented['mask']
    except Exception as e:
        augmented_images = image
        augmented_masks = mask

    return augmented_images, augmented_masks

def random_augmentation(sample, directory, new_directory):
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        file_path = os.path.join(directory, sample)
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]

        new_file_path = ''

        if np.any(mask != 0):

          aug_image, aug_mask = augment_image_and_mask(image, mask)

          new_file_name = f"{os.path.basename(file_path).replace('.h5', '')}_aug.h5"
          new_file_path = os.path.join(new_directory, new_file_name)
          with h5py.File(new_file_path, 'w') as new_file:
              new_file.create_dataset('image', data=aug_image)
              new_file.create_dataset('mask', data=aug_mask)

        return new_file_path

def mixup(image1, mask1, image2, mask2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)

    new_image = lam * image1 + (1 - lam) * image2
    new_mask = lam * mask1 + (1 - lam) * mask2

    return new_image, new_mask

def process_mixup(sample, directory, new_directory, files, alpha=0.4):
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    file_path = os.path.join(directory, sample)
    with h5py.File(file_path, 'r') as file:
        image = file['image'][()]
        mask = file['mask'][()]

    mixup_file_path = ''

    if np.any(mask != 0):

        mixup_index = np.random.randint(0, len(files))
        mixup_file_path = files[mixup_index]
        if mixup_file_path != file_path:
            with h5py.File(mixup_file_path, 'r') as mixup_file:
                mixup_image = mixup_file['image'][()]
                mixup_mask = mixup_file['mask'][()]

            mixup_image, mixup_mask = mixup(image, mask, mixup_image, mixup_mask, alpha=alpha)

            mixup_file_name = f"{os.path.basename(file_path).replace('.h5', '')}_mixup.h5"
            mixup_file_path = os.path.join(new_directory, mixup_file_name)
            with h5py.File(mixup_file_path, 'w') as mixup_file:
                mixup_file.create_dataset('image', data=mixup_image)
                mixup_file.create_dataset('mask', data=mixup_mask)

        return mixup_file_path

def cutmix(image1, mask1, image2, mask2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    lam = lam/4 if lam > 0.25 else lam

    Hx, Wx = image1.shape[0], image1.shape[1]

    Hm = int(np.sqrt(lam) * Hx)
    Wm = int(np.sqrt(lam) * Wx)

    center_top = (Hx - Hm) // 2
    center_left = (Wx - Wm) // 2

    top = np.random.randint(center_top, center_top + 10)
    left = np.random.randint(center_left, center_left + 10)

    image_Mask = np.zeros_like(image1, dtype=np.float32)
    image_Mask[top:top+Hm, left:left+Wm] = 1
    mask_Mask = np.zeros_like(mask1, dtype=np.float32)
    mask_Mask[top:top+Hm, left:left+Wm] = 1

    new_image = image_Mask * image2 + (1 - image_Mask) * image1
    new_mask = mask_Mask * mask2 + (1 - mask_Mask) * mask1

    return new_image, new_mask

def process_cutmix(sample, directory, new_directory, files, alpha=0.4):
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    file_path = os.path.join(directory, sample)
    with h5py.File(file_path, 'r') as file:
        image = file['image'][()]
        mask = file['mask'][()]

    cutmix_file_path = ''
    if np.any(mask != 0):

        mixup_index = np.random.randint(0, len(files))
        mixup_file_path = files[mixup_index]
        if mixup_file_path != file_path:
            with h5py.File(mixup_file_path, 'r') as mixup_file:
                mixup_image = mixup_file['image'][()]
                mixup_mask = mixup_file['mask'][()]

            cutmix_image, cutmix_mask = cutmix(image, mask, mixup_image, mixup_mask, alpha=alpha)

            cutmix_file_name = f"{os.path.basename(file_path).replace('.h5', '')}_cutmix.h5"
            cutmix_file_path = os.path.join(new_directory, cutmix_file_name)
            with h5py.File(cutmix_file_path, 'w') as cutmix_file:
                cutmix_file.create_dataset('image', data=cutmix_image)
                cutmix_file.create_dataset('mask', data=cutmix_mask)

        return cutmix_file_path


class BrainScanDataset(Dataset):
    def __init__(self, file_paths, normalization='None', deterministic=False):
        self.file_paths = file_paths
        if deterministic:
            np.random.seed(1)
        np.random.shuffle(self.file_paths)
        self.normalization = normalization

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]

            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))

            # Min-Max Normalization
            if self.normalization == "min-max":
              for i in range(image.shape[0]):
                  min_val = np.min(image[i])
                  image[i] = image[i] - min_val
                  max_val = np.max(image[i]) + 1e-4
                  image[i] = image[i] / max_val

            # Z-Score Normalization
            elif self.normalization == "z-score":
              image = (image - image.mean(axis=(1, 2), keepdims=True)) / (image.std(axis=(1, 2), keepdims=True) + 1e-5)

            # Percentile Normalization
            elif self.normalization == "percent":
              for i in range(image.shape[0]):
                  low_val = np.percentile(image[i], 2)
                  high_val = np.percentile(image[i], 98)
                  image[i] = np.clip(image[i], low_val, high_val)
                  image[i] = (image[i] - low_val) / (high_val - low_val + 1e-4)

            else: pass

            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask