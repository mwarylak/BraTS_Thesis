import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

now = datetime.now().strftime("%H:%M:%S")

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['text.color']       = '#474646'

def display_image_channels(image, title='Image Channels'):
    channel_names = ['T1-weighted (T1)', 'T1-weighted post contrast (T1c)', 'T2-weighted (T2)', 'Fluid Attenuated Inversion Recovery (FLAIR)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :]
        ax.imshow(channel_image, cmap='magma') #viridis
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.03)
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300) 
    plt.show()

def display_mask_channels_as_rgb(mask, title='Mask Channels as RGB'):
    channel_names = ['Necrotic (NEC)', 'Edema (ED)', 'Tumour (ET)']
    fig, axes = plt.subplots(1, 3, figsize=(9.75, 5))
    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, :, :] * 255
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300) 
    plt.show()

def overlay_masks_on_image(image, mask, title='Brain MRI with Tumour Masks Overlay'):
    t1_image = image[0, :, :]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300) 
    plt.show()

def overlay_masks(image, mask, alpha=0.5, title='Brain MRI with Tumour Masks Overlay'):
    t1_image = image[0, :, :]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)

    color_mask = np.zeros_like(rgb_image)
    color_mask[..., 0] = mask[0, :, :]
    color_mask[..., 1] = mask[1, :, :]
    color_mask[..., 2] = mask[2, :, :]

    rgb_image = (1 - alpha) * rgb_image + alpha * color_mask

    return rgb_image

def display_side_by_side(image, pred_mask, mask, title="Prediction Overlay"):
    overlay_pred = overlay_masks(image, pred_mask)
    overlay_mask = overlay_masks(image, mask)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(overlay_pred)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_mask)
    plt.title('Brain MRI with Tumour Masks Overlay', fontsize=18, y=1.02)
    plt.axis('off')  
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300)  
    plt.show()

def display_combined_mask_as_rgb(mask, title='Combined Mask as RGB'):
    rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

    # Przypisujemy kanały maski do odpowiednich kanałów koloru
    rgb_mask[..., 0] = mask[0, :, :] * 255  # Czerwony dla Necrotic (NEC)
    rgb_mask[..., 1] = mask[1, :, :] * 255  # Zielony dla Edema (ED)
    rgb_mask[..., 2] = mask[2, :, :] * 255  # Niebieski dla Tumour (ET)

    # Wyświetlanie obrazu
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_mask)
    plt.title(title, fontsize=20, y=1.02)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300)  
    plt.show()

def display_combined_predicted_mask_as_rgb(pred_mask, title='Predicted Combined Mask'):
    # Tworzymy obraz RGB, gdzie każda klasa jest innym kolorem
    rgb_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    rgb_pred_mask[pred_mask == 1] = [255, 0, 0]   # Czerwony dla NEC
    rgb_pred_mask[pred_mask == 2] = [0, 255, 0]   # Zielony dla ED
    rgb_pred_mask[pred_mask == 3] = [0, 0, 255]   # Niebieski dla ET

    # Wyświetlanie obrazu
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_pred_mask)
    plt.title(title, fontsize=20, y=1.02)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300)  
    plt.show()

def overlay_prediction(image, pred_mask, title='Prediction Overlay'):
    t1_image = image[0, :, :]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)

    rgb_pred = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.float32)
    rgb_pred[..., 0] = (pred_mask == 1).astype(np.float32)  # Czerwony dla NEC
    rgb_pred[..., 1] = (pred_mask == 2).astype(np.float32)  # Zielony dla ED
    rgb_pred[..., 2] = (pred_mask == 3).astype(np.float32)  # Niebieski dla ET

    overlay = np.clip(rgb_image + rgb_pred * 0.5, 0, 1)  # 0.5 = przezroczystość przewidywań

    return overlay

def overlay_real_masks(image, mask, title='Brain MRI with Tumour Masks Overlay'):
    t1_image = image[0, :, :]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)

    return rgb_image


def display_prediction_groundtruth(image, pred_mask, mask, title="Prediction Overlay"):
    overlay_pred = overlay_prediction(image, pred_mask)
    overlay_mask = overlay_real_masks(image, mask)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(overlay_pred)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_mask)
    plt.title('Brain MRI with Tumour Masks Overlay', fontsize=18, y=1.02)
    plt.axis('off')
    plt.savefig(f'/content/BraTS_Thesis/Images/{title}_{now}.png', bbox_inches='tight', dpi=300)  
    plt.show()