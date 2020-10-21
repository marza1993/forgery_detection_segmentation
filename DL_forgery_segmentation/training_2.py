import tensorflow as tf
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.models import load_model
from data_generator import data_loader
from segmentation_model import segmentation_model
import matplotlib.pyplot as plt
#from keras.metrics import MeanIoU
from tensorflow.keras.metrics import MeanIoU
from tensorflow import keras
from my_losses import jaccard_distance_loss, dice_coef_loss, dice_coef, weighted_loss
import numpy as np
from ImageLogger import ImageHistory
from segmentation_models import Unet
import os
from ModelOnLossImprCheckpoint import ModelOnLossImprCheckpoint

# menate dopo rbe
# prova tolto ambiente da git

#PATH_BASE = "D:\\dottorato\\copy_move\\DB_mio\\"
PATH_BASE = "D:\\dottorato\\copy_move\\DB_mio_small\\"

TRAIN_IMAGES_FOLDER = PATH_BASE + "train_images\\train\\"
TRAIN_MASKS_FOLDER = PATH_BASE + "train_masks\\train\\"
VAL_IMAGES_FOLDER = PATH_BASE + "val_images\\val\\"
VAL_MASKS_FOLDER = PATH_BASE + "val_masks\\val\\"
TEST_IMAGES_FOLDER = PATH_BASE + "test_images\\test\\"
TEST_MASKS_FOLDER = PATH_BASE + "test_masks\\test\\"
MODELS_PATH_BASE = PATH_BASE + "models\\"
WEIGHTS_FILE_NAME = "weights.hdf5"
MODEL_FILE_NAME = "model.hdf5"
LOSS_FILE_NAME = "best_loss.txt"

batch_size = 8
img_size = (256,256)



#tensorboard_log_dir = PATH_BASE + "log_esperimenti\\" + f"bs_{batch_size}_lr_{1.e-3}_new_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_log_dir = PATH_BASE + "log_esperimenti\\" + f"bs_{batch_size}_lr_{1.e-3}_resnet_imagenet_loaded_augm" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# creo l'oggetto per la gestione dinamica dei dati e per data-augmentation (training)
data_train = data_loader(batch_size, TRAIN_IMAGES_FOLDER, TRAIN_MASKS_FOLDER, img_size = img_size, apply_augmentation=False)

# oggetto per gestione dati in validation
data_val = data_loader(batch_size, VAL_IMAGES_FOLDER, VAL_MASKS_FOLDER, img_size = img_size, apply_augmentation=False)

print("N train/val samples: {}/{}".format(data_train.N_samples(), data_val.N_samples()))


# visualizzo alcuni esempi di immagine/groundtruth dal training set, con data-augmentation
resp = input("vuoi visualizzare alcune immagini/maschere di esempio da training set e validation set? [s/n]")

if resp == "s":
    print("*"*30)
    print("train examples:")
    print("*"*30)
    data_train.visualize_batch()
    print("*"*30)
    print("validation examples:")
    print("*"*30)
    data_val.visualize_batch()



model = None
model_path = None

resp = input("Vuoi riprendere il training di un modello esistente? [s/n]")


if resp == "s":

    model_path = input("directory del modello salvato: ")

    if model_path[-1] != "\\":
        model_path = model_path + "\\"

    # carico il modello 
    model = keras.models.load_model(model_path + MODEL_FILE_NAME, custom_objects = {'dice_coef_loss' : dice_coef_loss})

    # se il file con la migliore loss non esiste lo creo
    if not os.path.isfile(model_path + LOSS_FILE_NAME):
        with open(model_path + LOSS_FILE_NAME, 'w') as file_ref:
            file_ref.write('inf')

else:

    model_path = MODELS_PATH_BASE + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"
    weights_path = model_path + WEIGHTS_FILE_NAME

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # creo il file con la migliore loss e lo inizializzo a inf
    with open(model_path + LOSS_FILE_NAME, 'w') as file_ref:
        file_ref.write('inf')

    # creo il modello:
    #model = Unet('resnet34', input_shape=img_size + (3,), encoder_weights='imagenet')
    model = Unet('resnet18', input_shape=img_size + (3,), encoder_weights='imagenet')

    # compilo il modello
    mean_iou = MeanIoU(num_classes = 2, name = 'mean_iou')
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[mean_iou, 'accuracy'])



# stampo le informazioni sul modello costruito/caricato
print("output shape: {}".format(model.output_shape))
model.summary()
print("metrics:")
print(model.metrics_names)  


N_epochs = 50

# creo le callback per il training
checkpoint = ModelOnLossImprCheckpoint(model_path)
early_stopping = EarlyStopping(monitor='val_mean_iou', mode='max', verbose=1, patience=N_epochs)
#reduceLR = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1, update_freq = 'batch', profile_batch = 0)
image_logger = ImageHistory(tensorboard_log_dir, data_val,  data_train.N_samples() // batch_size, N_imgs=5)

callbacks_list = [checkpoint, early_stopping, tensorboard_callback, image_logger]

# effettuo il training
history = model.fit(
            data_train,
            steps_per_epoch = data_train.N_samples() // batch_size,
            epochs = N_epochs, 
            validation_data = data_val, 
            validation_steps = data_val.N_samples() // batch_size,
            callbacks=callbacks_list)



