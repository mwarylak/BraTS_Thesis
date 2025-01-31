import display_methods_NN
import os 
import h5py

directory = "/home/magda/Pulpit/BraTS_Thesis/Files/Test_Samples"
files = [f for f in os.listdir(directory) if f.endswith('.h5')]


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


image, mask = sample_h5_file(directory, files, 8)

display_methods_NN.display_image_channels(image, title='Image Channels')
display_methods_NN.display_mask_channels_as_rgb(mask, title='Ground Truth Mask Channels')
display_methods_NN.display_combined_mask_as_rgb(mask, title='Combined Mask')
display_methods_NN.overlay_masks_on_image(image, mask, title='Brain MRI with Tumour Masks Overlay')

