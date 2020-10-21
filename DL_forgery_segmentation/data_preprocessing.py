import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

INPUT_PATH = "D:\\dottorato\\copy_move\\MICC-F600\\"
OUTPUT_PATH = "D:\\dottorato\\copy_move\\MICC-F600_DL_segmentation\\"
IMAGES_FOLDER = OUTPUT_PATH + "images\\"
MASKS_FOLDER = OUTPUT_PATH + "masks\\"


# ************************************************************************************
# carico tutte le immagini forged e le rispettive mappe con il groundtruth.
# Per farlo leggo tutti i file che terminano con "_gt.png" o "_gt.jpg", che corrispondono ai groundtruth e
# le corrispondenti immagini che hanno lo stesso nome ma senza "_gt".
regex_groundtruth = r".*_gt\.(png|jpg)$"

forged_masks_list = [f for f in os.listdir(INPUT_PATH) if re.search(regex_groundtruth,f)]
forged_images_list = [f.replace("_gt.", ".") for f in forged_masks_list]

print("n. forged images found: {}".format(len(forged_images_list)))
print("n. corresponding masks found: {}".format(len(forged_masks_list)))

# leggo le immagini, binarizzo la mappa e salvo nella percorso di output
n = 0
for img_name,mask_name in zip(forged_images_list, forged_masks_list):

    print("immagine {}:".format(n))
    #print("{}: {} - {}".format(n+1, img_name, mask_name))
    img = cv2.imread(INPUT_PATH + img_name)
    mask = cv2.imread(INPUT_PATH + mask_name, cv2.IMREAD_GRAYSCALE)

    #print("img.shape: {}, mask.shape: {}".format(img.shape, mask.shape))

    # binarizzazione della maschera
    ret,binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    print("binary.shape: {}".format(binary.shape))

    estensione_img = img_name.split(".")[-1]
    estensione_mask = mask_name.split(".")[-1]
    # salvo nel nuovo percorso
    cv2.imwrite(IMAGES_FOLDER + str(n) + "." + estensione_img, img)
    #cv2.imwrite(MASKS_FOLDER + str(n) + "_mask." + estensione_mask, binary)
    # devo avere lo stesso nome per la maschera, in modo da poter fare il match con i generatori di keras
    cv2.imwrite(MASKS_FOLDER + str(n) + "." + estensione_img, binary)

    n+=1


 
# ************************************************************************************
# carico ora tutte le immagini originali e genero la corrispondente maschera grountruth (tutta nera)

# espressione regolare per ottenere tutti i nomi di immagini che non contengono "_gt" prima dell'estensione.
regex_images_not_gt = r".*(?<!_gt)\.(png|jpg)$"

# le immagini originali sono quelle ottenute da questa regex che non appartengono all'insieme delle immagini forged
images_list = [f for f in os.listdir(INPUT_PATH) if re.search(regex_images_not_gt,f)]
print("tot immagini: {}".format(len(images_list)))
original_images_list = [f for f in images_list if not f in forged_images_list]
print("tot original images: {}".format(len(original_images_list)))

for img_name in original_images_list:

    print("immagine {}:".format(n))
    img = cv2.imread(INPUT_PATH + img_name)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype = 'uint8')
    print("mask.shape: {}".format(mask.shape))
    estensione_img = img_name.split(".")[-1]
    # salvo nel nuovo percorso
    cv2.imwrite(IMAGES_FOLDER + str(n) + "." + estensione_img, img)
    #cv2.imwrite(MASKS_FOLDER + str(n) + "_mask." + estensione_mask, mask)
    cv2.imwrite(MASKS_FOLDER + str(n) + "." + estensione_img, mask)
    
    n+=1


# ************************************************************************************
# effettuo lo split dei dati in training, validation e test set

TRAIN_IMAGES_FOLDER = OUTPUT_PATH + "train_images\\train\\"
TRAIN_MASKS_FOLDER = OUTPUT_PATH + "train_masks\\train\\"
VAL_IMAGES_FOLDER = OUTPUT_PATH + "val_images\\val\\"
VAL_MASKS_FOLDER = OUTPUT_PATH + "val_masks\\val\\"
TEST_IMAGES_FOLDER = OUTPUT_PATH + "test_images\\test\\"
TEST_MASKS_FOLDER = OUTPUT_PATH + "test_masks\\test\\"


# carico le immagini dalle cartelle e le ordino in base al numero contenuto nel nome del file
image_list = [f for f in os.listdir(IMAGES_FOLDER)]
image_list.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
mask_list = [f for f in os.listdir(MASKS_FOLDER)]
mask_list.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
print("n. image found: {}, n. masks found: {}".format(len(image_list), len(mask_list)))

indici = np.arange(0, len(image_list))

# mescolo
random.seed(100)
random.shuffle(indici)

image_list = [image_list[i] for i in indici]
mask_list = [mask_list[i] for i in indici]


# creazione del training set: 80% delle immagini
TRAIN_PERC_SPLIT = int(0.8 * len(image_list))
train_images = image_list[:TRAIN_PERC_SPLIT]
train_masks = mask_list[:TRAIN_PERC_SPLIT]

# creazione del validation set: 20% del training set
VAL_PERC_SPLIT = int(0.2 * len(train_images))
val_images = train_images[:VAL_PERC_SPLIT]
val_masks = train_masks[:VAL_PERC_SPLIT]

train_images = train_images[VAL_PERC_SPLIT:]
train_masks = train_masks[VAL_PERC_SPLIT:]

# creazione del test set: 20% delle immagini
test_images = image_list[TRAIN_PERC_SPLIT:]
test_masks = mask_list[TRAIN_PERC_SPLIT:]


print("train images: {}, validation images: {}, test images: {}".format(len(train_images), len(val_images), len(test_images)))
print("train masks: {}, validation masks: {}, test masks: {}".format(len(train_masks), len(val_masks), len(test_masks)))

# sposto le immagini nelle cartelle apposite
for i in range(0,len(train_images)):
    os.rename(IMAGES_FOLDER + train_images[i], TRAIN_IMAGES_FOLDER + train_images[i])
    os.rename(MASKS_FOLDER + train_masks[i], TRAIN_MASKS_FOLDER + train_masks[i])

for i in range(0,len(val_images)):
    os.rename(IMAGES_FOLDER + val_images[i], VAL_IMAGES_FOLDER + val_images[i])
    os.rename(MASKS_FOLDER + val_masks[i], VAL_MASKS_FOLDER + val_masks[i])

for i in range(0,len(test_images)):
    os.rename(IMAGES_FOLDER + test_images[i], TEST_IMAGES_FOLDER + test_images[i])
    os.rename(MASKS_FOLDER + test_masks[i], TEST_MASKS_FOLDER + test_masks[i])



