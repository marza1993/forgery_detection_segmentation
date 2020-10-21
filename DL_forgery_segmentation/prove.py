#import tensorflow as tf
#import datetime
#from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from keras.callbacks import TensorBoard
#from keras.models import load_model
#from data_generator import data_loader
#from segmentation_model import segmentation_model
#import matplotlib.pyplot as plt
#from keras.metrics import MeanIoU
#from my_losses import jaccard_distance_loss, dice_coef_loss, dice_coef, weighted_loss
#import pickle
#import numpy as np
#from keras.optimizers import Adam
#from ImageLogger import ImageHistory
#from segmentation_models import Unet
#import keras

#PATH_BASE = "D:\\dottorato\\copy_move\\DB_mio_small\\"

#TRAIN_IMAGES_FOLDER = PATH_BASE + "train_images\\train\\"
#TRAIN_MASKS_FOLDER = PATH_BASE + "train_masks\\train\\"
#VAL_IMAGES_FOLDER = PATH_BASE + "val_images\\val\\"
#VAL_MASKS_FOLDER = PATH_BASE + "val_masks\\val\\"
#model_path = PATH_BASE + "models\\"

#batch_size = 8
#img_size = (256,256)


## creo l'oggetto per la gestione dinamica dei dati e per data-augmentation (training)
##data_train = data_loader(batch_size, TRAIN_IMAGES_FOLDER, TRAIN_MASKS_FOLDER)
#data_train = data_loader(batch_size, TRAIN_IMAGES_FOLDER, TRAIN_MASKS_FOLDER, img_size = img_size, apply_augmentation=False)

## oggetto per gestione dati in validation
#data_val = data_loader(batch_size, VAL_IMAGES_FOLDER, VAL_MASKS_FOLDER, img_size = img_size, apply_augmentation=False)


#model = Unet('resnet34', input_shape=img_size + (3,), encoder_weights='imagenet')

#mean_iou = MeanIoU(num_classes = 2, name = 'mean_iou')

#model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[mean_iou, 'accuracy'])


#N_epochs = 5


#history = model.fit(
#            data_train,
#            steps_per_epoch = data_train.N_samples() // batch_size,
#            epochs = N_epochs, 
#            validation_data = data_val, 
#            validation_steps = data_val.N_samples() // batch_size
#            )

## Calling `save('my_model')` creates a SavedModel folder `my_model`.
#model.save("my_model")

## It can be used to reconstruct the model identically.
#reconstructed_model = tf.keras.models.load_model("my_model", custom_objects={'dice_coef_loss':dice_coef_loss, 'mean_iou':mean_iou})

## Let's check:
#np.testing.assert_allclose(
#    model.predict(test_input), reconstructed_model.predict(test_input)
#)


#history = reconstructed_model.fit(
#            data_train,
#            steps_per_epoch = data_train.N_samples() // batch_size,
#            epochs = N_epochs, 
#            validation_data = data_val, 
#            validation_steps = data_val.N_samples() // batch_size
#            )


## Plot training & validation accuracy values
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'validation'], loc='best')
#plt.show()

## Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.legend(['Train', 'validation'], loc='best')
#plt.show()

# ********************************************************


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU

np.random.seed(51)

from my_losses import dice_coef_loss

metrica = MeanIoU(num_classes = 2, name = 'mean_iou')


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=dice_coef_loss, metrics = [metrica, 'accuracy'])
    return model


model = get_model()

# Train the model.
test_input = np.random.rand(128, 32)
test_target = np.random.rand(128, 1)
model.fit(test_input, test_target, epochs = 30)


nome = 'menate.hdf5'

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save(nome)

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model(nome)

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target, epochs = 100)
