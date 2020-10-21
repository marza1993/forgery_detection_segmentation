from tensorflow import keras
from data_generator import data_loader
from segmentation_model import segmentation_model
import matplotlib.pyplot as plt
from my_losses import dice_coef_loss
import os
import re
from keras.metrics import MeanIoU
import numpy as np
import random


PATH_BASE = "D:\\dottorato\\copy_move\\DB_mio\\"
#TEST_IMAGES_FOLDER = PATH_BASE + "val_images\\val\\"
#TEST_MASKS_FOLDER = PATH_BASE + "val_masks\\val\\"
#TEST_IMAGES_FOLDER = PATH_BASE + "train_images\\train\\"
#TEST_MASKS_FOLDER = PATH_BASE + "train_masks\\train\\"
TEST_IMAGES_FOLDER = PATH_BASE + "test_images\\test\\"
TEST_MASKS_FOLDER = PATH_BASE + "test_masks\\test\\"
MODEL_FILE_NAME = "model.hdf5"


target_W = 256
target_H = 256
batch_size = 16


data_test = data_loader(batch_size, TEST_IMAGES_FOLDER, TEST_MASKS_FOLDER, img_size = (target_H, target_W), apply_augmentation=False)


# carico e compilo il modello della rete
model = None
model_path = input("directory del modello salvato: ")
if model_path[-1] != "\\":
    model_path = model_path + "\\"
    model = keras.models.load_model(model_path + MODEL_FILE_NAME, custom_objects = {'dice_coef_loss' : dice_coef_loss})



# effettuo alcune predizioni e visualizzo il risultato qualitativo
print("*"*30)
print("predizione su immagini test")
print("*"*30)


def visualize_predictions_on_random_batch(data_generator):

    N_batches = data_generator.__len__()

    batch_img, batch_mask = data_generator.__getitem__(int(np.random.randint(0, N_batches, 1)))

    print("batch_img.shape: {}".format(batch_img.shape))
    print("batch_mask.shape: {}".format(batch_mask.shape))

    # effettuo la predizione su un batch
    batch_size = len(batch_img)
    predictions = model.predict(batch_img, batch_size)
    print("prediction.shape: {}".format(predictions.shape))

    

    for i in range(0,batch_size):

        # effettuo la predizione

        f,arraxis = plt.subplots(1,3)
        arraxis[0].imshow(batch_img[i])
        arraxis[0].set_title('image')
        arraxis[1].imshow(batch_mask[i,...,0])
        arraxis[1].set_title('groundtruth')
        arraxis[2].imshow(predictions[i,...,0])
        arraxis[2].set_title('prediction')
        plt.show()

        nonz = np.unique(predictions[i])
        print("non zero element of hist: {}".format(len(nonz)))
        print(nonz)
        print("min/max img vals: {}/{} ".format(np.min(batch_img[i]), np.max(batch_img[i])))
        print("min/max mask vals: {}/{} ".format(np.min(batch_mask[i,:,:,:]), np.max(batch_mask[i,:,:,:])))
        print("min/max pred vals: {}/{} ".format(np.min(predictions[i,:,:,:]), np.max(predictions[i,:,:,:])))

        #print("valori:")
        #print(predictions[i])



visualize_predictions_on_random_batch(data_test)

# valuto il modello sul test set
results = model.evaluate_generator(data_test)
print("len(results): {}".format(len(results)))
print("dice_coef_loss: {}".format(results[0]))
print("meanIOU: {}".format(results[1]))
print("accuracy: {}".format(results[2]))
