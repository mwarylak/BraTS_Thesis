import matplotlib.pyplot as plt
import numpy as np

def display_image_channels(image, title='Image Channels'):
    channel_names = ['T1-weighted (T1)', 'T1-weighted post contrast (T1c)', 'T2-weighted (T2)', 'Fluid Attenuated Inversion Recovery (FLAIR)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :]
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.03)
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

    plt.show()