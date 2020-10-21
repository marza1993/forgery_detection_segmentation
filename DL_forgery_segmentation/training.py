from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from data_generator import data_generator
from segmentation_model import segmentation_model
import matplotlib.pyplot as plt
from keras.metrics import MeanIoU
from my_losses import jaccard_distance_loss
import pickle


PATH_BASE = "D:\\dottorato\\copy_move\\MICC-F600_DL_segmentation\\"

TRAIN_IMAGES_FOLDER = PATH_BASE + "train_images\\"
TRAIN_MASKS_FOLDER = PATH_BASE + "train_masks\\"
VAL_IMAGES_FOLDER = PATH_BASE + "val_images\\"
VAL_MASKS_FOLDER = PATH_BASE + "val_masks\\"
TEST_IMAGES_FOLDER = PATH_BASE + "test_images\\"
TEST_MASKS_FOLDER = PATH_BASE + "test_masks\\"


#N_train_samples = 384
#N_val_samples = 96
N_train_samples = 102
N_val_samples = 24
batch_size = 2

data = data_generator(
        TRAIN_IMAGES_FOLDER, 
        TRAIN_MASKS_FOLDER, 
        VAL_IMAGES_FOLDER, 
        VAL_MASKS_FOLDER, 
        TEST_IMAGES_FOLDER, 
        TEST_MASKS_FOLDER, batch_size)


# funzione per visualizzare delle coppie immagine/maschera tramite un generatore (che eventualmente
# potrebbe effettuare data-augmentation
def visualize_image_masks_examples(image_mask_generator, N_batches):

    for j in range(0, N_batches):
        batch_image, batch_mask = next(image_mask_generator)
        print("bath_image.shape: {}".format(batch_image.shape))

        for i in range(0, batch_image.shape[0]):
            f,arraxis = plt.subplots(1,2)
            arraxis[0].imshow(batch_image[i])
            arraxis[1].imshow(batch_mask[i])
            plt.show()



train_gen = data.train_image_mask_generator()
val_gen = data.val_image_mask_generator()

#visualize_image_masks_examples(train_gen, 1)
#visualize_image_masks_examples(val_gen, 1)


# costruisco il modello per la segmentazione
model = segmentation_model.build_unet()

# compilo il modello con la loss e la metrica da monitorare

#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [MeanIoU(num_classes = 2, name = 'mean_iou')])
#model.compile(optimizer = 'sgd', loss = 'mse', metrics = [MeanIoU(num_classes = 2, name = 'mean_iou')])
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = 'adam', loss = jaccard_distance_loss, metrics = [MeanIoU(num_classes = 2, name = 'mean_iou')])

#model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics = [MeanIoU(num_classes = 2, name = 'mean_iou')])

model.summary()


# imposto la callback per salvare i pesi migliori
model_path = PATH_BASE + "\\models\\"
checkpoint = ModelCheckpoint(model_path + "best_segmentation_model.hdf5", monitor='val_mean_iou', verbose=1, save_best_only=True, mode='max')
#checkpoint = ModelCheckpoint(model_path + "best_segmentation_model_val.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

N_epochs = 200

# callback per early stopping quando l'accuracy non sale per più di N epoche
#early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=N_epochs/10)
callbacks_list = [checkpoint, early_stopping]


history = model.fit_generator(
            data.train_image_mask_generator(), 
            steps_per_epoch = N_train_samples // batch_size,
            epochs=N_epochs,
            validation_data=data.val_image_mask_generator(), 
            validation_steps=N_val_samples // batch_size, 
            callbacks=callbacks_list)

print(history.history.keys())

with open(model_path + "trainHistoryDict", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

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


# ------------- evaluation ------------------
#N_test_samples = 80

## carico i pesi migliori salvati
#weight_file_name = "best_model_acc_0.977.hdf5"
#model = binary_classification_model.build_and_compile(input_channels = 3)

#model.load_weights(model_path + weight_file_name)

## valuto l'accuratezza sul test set
#score = model.evaluate_generator(data.get_test_generator(), steps = N_test_samples // batch_size)
##score = model.evaluate(x_test, y_test, verbose=0)
#print("score: {}".format(score))