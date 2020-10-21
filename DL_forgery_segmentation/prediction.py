from data_generator import data_loader
from segmentation_model import segmentation_model
import matplotlib.pyplot as plt
from my_losses import jaccard_distance_loss
import cv2
import os
import re
from keras.metrics import MeanIoU
import numpy as np
import random

target_W = 256
target_H = 256
batch_size = 8

#PATH_BASE = "D:\\dottorato\\copy_move\\MICC-F600_DL_segmentation\\"

PATH_BASE = "D:\\dottorato\\copy_move\\DB_mio\\"


#TEST_IMAGES_FOLDER = PATH_BASE + "test_images\\test\\"
#TEST_MASKS_FOLDER = PATH_BASE + "test_masks\\test\\"


#TEST_IMAGES_FOLDER = PATH_BASE + "val_images_small\\val\\"
#TEST_MASKS_FOLDER = PATH_BASE + "val_masks_small\\val\\"


TEST_IMAGES_FOLDER = PATH_BASE + "val_images\\val\\"
TEST_MASKS_FOLDER = PATH_BASE + "val_masks\\val\\"
model_path = PATH_BASE + "models\\"


#test_images_folder = path_base + "train_images\\train\\"
#test_masks_folder = path_base + "train_masks\\train\\"



# carico un po' di immagini di test
regex_img = r".*\.(png|jpg)$"

list_imgs = [f for f in os.listdir(TEST_IMAGES_FOLDER) if re.search(regex_img,f)]
list_masks = [f for f in os.listdir(TEST_MASKS_FOLDER) if re.search(regex_img,f)]


print("n. immagini test: {}".format(len(list_imgs)))
print("n. maschere corrispondenti: {}".format(len(list_masks)))


# carico le immagini
imgs = []
for i in range(0,len(list_imgs)):
    #print("immagine {}".format(i))
    img = cv2.cvtColor(cv2.imread(TEST_IMAGES_FOLDER + list_imgs[i]), cv2.COLOR_BGR2RGB) 
    imgs.append(img)
imgs = np.array(imgs)
print("imgs.shape: {}".format(imgs.shape))


# carico le maschere corrispondenti
masks = []
for i in range(0,len(list_masks)):
    #print("maschera {}".format(i))
    mask = cv2.imread(TEST_MASKS_FOLDER + list_masks[i], cv2.IMREAD_GRAYSCALE) 
    masks.append(mask)
masks = np.array(masks)
print("masks.shape: {}".format(masks.shape))




#weight_file_name = "best_segmentation_model.hdf5"
#weight_file_name = "best_segmentation_model_custom_data_gen.hdf5"
#weight_file_name = "custom_data_gen_small.hdf5"
weight_file_name = "best_dice_coef_no_aug.hdf5"
model = segmentation_model.build_unet(pretrained_weights = model_path + weight_file_name, input_size=((target_W, target_W)+(3,)))

#model.compile(optimizer = 'adam', loss = jaccard_distance_loss, metrics = [MeanIoU(num_classes = 2, name = 'mean_iou')])

#model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics = [MeanIoU(num_classes = 2, name = 'mean_iou'), 'accuracy'])

model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[MeanIoU(num_classes = 2, name = 'mean_iou'), 'accuracy'])


# effettuo alcune predizioni e visualizzo il risultato
print("*"*30)
print("predizione su immagini test")
print("*"*30)




def visualize_predictions(imgs, masks, target_W, target_H, model, show = True):

    indici = np.arange(0, len(imgs))

    # mescolo
    random.seed(102)
    random.shuffle(indici)

    for i in range(0,len(imgs)):

        print("predizione {}".format(i))
        resized_img = cv2.resize(imgs[indici[i]], (target_W,target_H), interpolation = cv2.INTER_NEAREST)
        resized_mask = cv2.resize(masks[indici[i]], (target_W,target_H), interpolation = cv2.INTER_NEAREST)
        prediction = model.predict(np.expand_dims(resized_img, axis = 0))
        print("prediction.shape: {}".format(prediction.shape))


        f,axis = plt.subplots(1,3)
        axis[0].imshow(resized_img)
        axis[0].set_title('image')
        axis[1].imshow(resized_mask)
        axis[1].set_title('groundtruth')
        axis[2].imshow(prediction[0,...,0])
        axis[2].set_title('prediction')
       

        #print("min: {}, max: {}, mean: {}, std: {}".format(np.min(prediction[0,:,:,1]), np.max(prediction[0,:,:,1]), np.mean(prediction[0,:,:,1]),
        #                                                                                                            np.std(prediction[0,:,:,1])
        #                                                                                                            ))
        plt.show()


        #if show:
        #    plt.show()
        #else:
        #    plt.savefig(PATH_BASE + "prediction_test\\new\\prediction_" + list_imgs[indici[i]])

visualize_predictions(imgs, masks, target_W, target_H, model)