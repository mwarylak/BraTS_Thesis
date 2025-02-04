# BraTS_Thesis

The prepared repository allows to train machine learning algorithms and convolutional neural networks for the task of segmenting semantic brain tumor lesions. The BraTS2020 collection was used to train and validate the models. 

To start training the neural network model, it is enough to clone the repository (Google Colaboratory is the recommended environment using the command !git clone https://github.com/mwarylak/BraTS_Thesis.git) and run successive BraTS_Segmentation.ipynb notepad cells up to the cell with the train_NN.py script. It is also necessary to complete the NN_config.json file, in which the learning parameters are set. Among the parameters are available:
- device - the device on which the training will be performed
- n_epochs - number of training epochs 
- batch_size 
- learning_rate 
- batches_per_epoch
- lr_decay_factor - learning rate reduction value 
- normalization_type - type of data normalization, available types are:

  1.  "min-max"
  2.  "z-score"
  3.  "percent"

- augmentation_types - possible types of data augmentation such as NEC_ET, random, mixup lub cutmix. This variable is in the form of a list, so it is possible to use several types of augmentation
- model_saving - possibility to save the trained model
- loss_function - loss function used in training - possible loss functions are:

  1.  "BCE"
  2.  "Dice"
  3.  "Combined"

- activation_fun - neural activation function, selectable ReLU or LeakyReLU
- model - neural network model such as: UNet, AttentionUNet, SeprateUNet, InverseUNet, ResNetUNet, SpatialUNet, ChannelUNet

It is also possible to perform tests on one of the three prepared neural network models - UNet, AttentionUNet and UNet with ResNet18 encoder. To do this, you need to run the cell with the test_NN.py script and properly transform the NN_config.json file. 
For the UNet model, the necessary settings are:
  -  “device": “cuda”,
  -  “batch_size": 64,
  -   “learning_rate": 0.001,
  - “batches_per_epoch": 64,
  -  “loss_function": “BCE”,
  - “activation_fun": “LeakyReLU”,
  - “model": “UNet”.

For AttentionUNet:
  -  “device": “cuda”,
  -  “batch_size": 64,
  - “learning_rate": 0.001,
  - “batches_per_epoch": 64,
  - “loss_function": “BCE”,
  - “activation_fun": “ReLU”,
  -  “model": “AttentionUNet”

For UNet with ResNet:
  -  “device": “cuda”,
  -  “batch_size": 64,
  -  “learning_rate": 0.001,
  -  “batches_per_epoch": 64,
  -  “loss_function": “BCE”,
  -  “activation_fun": “LeakyReLU”,
  -  “model": "ResNetUNet"

Implementation of all models and their training and testing are also possible through the BraTS_Segmentation_NN.ipynb notebook. However, it requires running more consecutive cells and attaching the corresponding files from the GitHub repository.

For training and testing of Random Forest and SVM models, also the BraTS_Segmentation.ipynb notebook is used. To run training or testing of the corresponding models also needs to complete the configuration file ML_config.json by selecting the name of the model to train and then run the cell with the script train_ML.py or test_ML.py.
In the case of the Random Forest algorithm, there is no way to test it - the trained model is too large to store in a git repository. First you need to do the training and then use the trained model for testing.