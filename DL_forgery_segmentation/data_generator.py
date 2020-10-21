import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
import keras
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt

class data_generator(object):
    """
    permette di configurare il percorso dei dati per il caricamento runtime ed eventuale data augmentation,
    tramite i data generator di keras.
    precondizione è che i dati siano organizzati in un path del tipo:
        data -> | train_images  |   train   !
                | train_masks   |   train   |
                | val_images    |   val     |
                | val_masks     |   val     |
                | test_images   |   test    |
                | test_masks    |   test    |

    (vedi data_preprocessing.py)
    questo permette di utilizzare i data generator di keras di default                                                            
    """

    def __init__(self, train_image_path, train_masks_path, 
                 val_image_path, val_masks_path, 
                 test_image_path, test_masks_path,
                 batch_size = 16,
                 target_width = 512,
                 target_height = 512):

        self.train_image_path = train_image_path
        self.train_masks_path = train_masks_path
        self.val_image_path = val_image_path
        self.val_masks_path = val_masks_path
        self.test_image_path = test_image_path
        self.test_masks_path = test_masks_path
        self.batch_size = batch_size
        self.target_width = target_width
        self.target_height = target_height


        self.SEED = 100

        # this is the augmentation configuration we will use for training
        self.train_datagen = ImageDataGenerator(
                samplewise_center=True,
                rescale=1./255,
                shear_range=0.5,
                zoom_range=0.5,
                rotation_range=20,
                width_shift_range=0.5,
                #height_shift_range=0.5,
                horizontal_flip=True,
                vertical_flip=False,
                )


        self.test_datagen = ImageDataGenerator(rescale=1./255)


        self.train_image_generator = self.train_datagen.flow_from_directory(
                self.train_image_path,
                target_size=(target_width, target_height),
                class_mode=None,    # siccome è segmentazione, non classificazione
                batch_size=batch_size,
                seed = self.SEED)

        self.train_mask_generator = self.train_datagen.flow_from_directory(
                self.train_masks_path,
                target_size=(target_width, target_height),
                color_mode='grayscale',
                class_mode=None,
                batch_size=batch_size,
                seed = self.SEED)

        self.val_image_generator = self.test_datagen.flow_from_directory(
                self.val_image_path,
                target_size=(target_width, target_height),
                class_mode=None,
                batch_size=batch_size,
                seed = self.SEED)

        self.val_mask_generator = self.test_datagen.flow_from_directory(
                self.val_masks_path,
                target_size=(target_width, target_height),
                color_mode='grayscale',
                class_mode=None,
                batch_size=batch_size,
                seed = self.SEED)

        self.test_image_generator = self.test_datagen.flow_from_directory(
                self.test_image_path,
                target_size=(target_width, target_height),
                class_mode=None,
                batch_size=batch_size,
                seed = self.SEED)

        self.test_mask_generator = self.test_datagen.flow_from_directory(
                self.test_masks_path,
                target_size=(target_width, target_height),
                color_mode='grayscale',
                class_mode=None,
                batch_size=batch_size,
                seed = self.SEED)


        #self.test_generator = self.test_datagen.flow_from_directory(
        #        self.test_path,  
        #        target_size=(150, 150),
        #        batch_size=batch_size)

        self.train_generator = zip(self.train_image_generator, self.train_mask_generator)
        self.val_generator = zip(self.val_image_generator, self.val_mask_generator)
        self.test_generator = zip(self.test_image_generator, self.test_mask_generator)


    def train_image_mask_generator(self):
        for (img, mask) in self.train_generator:
            yield (img, mask)

    def val_image_mask_generator(self):
        for (img, mask) in self.val_generator:
            yield (img, mask)

    def test_image_mask_generator(self):
        for (img, mask) in self.test_generator:
            yield (img, mask)

    def get_train_generator(self):

        return self.train_generator

    def get_val_generator(self):
        return self.val_generator

    #def get_test_generator(self):
    #    return self.test_generator




class data_loader(tf.keras.utils.Sequence):

    """Helper to iterate over the data (as Numpy arrays)."""



    def __init__(self, batch_size, image_path, mask_path, img_size = (512,512), apply_augmentation = True):

        self.batch_size = batch_size
        self.img_size = img_size

        self.image_path = image_path
        self.mask_path = mask_path

        # se questo flag è false l'immagine e la maschera vengono solo riscalati
        self.apply_augmentation = apply_augmentation

        # carico la lista dei nomi delle immagini nel percorso passato
        regex_img = r".*\.(png|jpg)$"

        self.list_imgs = [f for f in os.listdir(image_path) if re.search(regex_img,f)]
        self.list_imgs.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
        self.list_masks = [f for f in os.listdir(mask_path) if re.search(regex_img,f)]
        self.list_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])

        print("n. imgs: {}, n. masks: {}".format(len(self.list_imgs), len(self.list_masks)))


        # mescolo in maniera coerente le immagini e le corrispondenti maschere:
        coppie = list(zip(self.list_imgs, self.list_masks))

        random.seed(100)
        random.shuffle(coppie)

        coppie = np.array(coppie)
        #print("coppie.shape: {}".format(coppie.shape))
        self.list_imgs = list(coppie[:,0])
        self.list_masks = list(coppie[:,1])

        # per la data augmentation
        self.img_augmenter = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
                rescale=1./255,
                #shear_range=0.5,
                #zoom_range=0.1,
                #rotation_range=60,
                #rotation_range=10,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                #vertical_flip=True,
                vertical_flip=False,
                fill_mode = 'wrap'
                )

        self.mask_augmenter = ImageDataGenerator(
                rescale=1./255,
                #shear_range=0.5,
                #zoom_range=0.1,
                #rotation_range=60,
                #rotation_range=10,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                #vertical_flip=True,
                vertical_flip=False,
                fill_mode = 'wrap'
            )

        self.rescaler = ImageDataGenerator(
                rescale=1./255,
                #samplewise_center=True,
                
                #samplewise_std_normalization=True
                )

    
    def N_samples(self):
        return len(self.list_imgs)


        #print("*"*30)
        #print("lista immagini: ")
        #print(self.list_imgs)
        #print("*"*30)
        #print("lista maschere: ")
        #print(self.list_masks)


    def __len__(self):
        return len(self.list_imgs) // self.batch_size


    #def __getitem__(self, idx):

        
    #    """Returns tuple (input_img, mask) correspond to batch #idx."""
    #    i = idx * self.batch_size
    #    batch_img_list = self.list_imgs[i : i + self.batch_size]
    #    batch_mask_list = self.list_masks[i : i + self.batch_size]

    #    x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
    #    y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")

    #    for i in range(self.batch_size):


    #        data_manipulator = self.img_augmenter if self.apply_augmentation else self.rescaler

    #        # applico una trasformazione random all'immagine e alla maschera
    #        params_img_transf = data_manipulator.get_random_transform(x[i].shape)

    #        # carico l'immagine
    #        img = load_img(self.image_path + batch_img_list[i], target_size=self.img_size)
    #        x[i] = np.array(img)# / 255.0

    #        # carico la maschera corrispondente
    #        img = load_img(self.mask_path + batch_mask_list[i], target_size=self.img_size, color_mode="grayscale")
    #        #y[i] = np.expand_dims(np.array(img) / 255.0, 2)
    #        y[i] = np.expand_dims(np.array(img), 2)


    #        x[i] = data_manipulator.apply_transform(data_manipulator.standardize(x[i]), params_img_transf)
    #        y[i] = data_manipulator.apply_transform(data_manipulator.standardize(y[i]), params_img_transf)

    #        # binarizzo 
    #        y[y<0.5] = 0.0
    #        y[y>=0.5] = 1.0

            
    #    return x, y


    def __getitem__(self, idx):

        
        """Returns tuple (input_img, mask) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_img_list = self.list_imgs[i : i + self.batch_size]
        batch_mask_list = self.list_masks[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")

        for i in range(self.batch_size):


            data_manipulator = self.img_augmenter if self.apply_augmentation else self.rescaler
            data_mask_manipulator = self.mask_augmenter if self.apply_augmentation else self.rescaler

            # applico una trasformazione random all'immagine e alla maschera

            random_seed = int(np.random.randint(0, 1000, 1))

            params_img_transf = data_manipulator.get_random_transform(x[i].shape, seed = random_seed)
            params_mask_transf = data_mask_manipulator.get_random_transform(y[i].shape, seed = random_seed)

            #print("img trans params: {}".format(params_img_transf))
            #print("mask trans params: {}".format(params_mask_transf))

            # carico l'immagine
            img = load_img(self.image_path + batch_img_list[i], target_size=self.img_size)
            x[i] = np.array(img)# / 255.0

            # carico la maschera corrispondente
            mask = load_img(self.mask_path + batch_mask_list[i], target_size=self.img_size, color_mode="grayscale")
            mask = np.array(mask, dtype='float32')
            #print("mask.shape: {}".format(mask.shape))

            x[i] = data_manipulator.apply_transform(data_manipulator.standardize(x[i]), params_img_transf)

            #mask = data_manipulator.apply_transform(data_manipulator.standardize(mask), params_img_transf)

            mask = data_mask_manipulator.apply_transform(data_mask_manipulator.standardize(np.expand_dims(mask,2)), params_mask_transf)
            #print("mask.shape: {}".format(mask.shape))


            y[i,...] = np.around(mask)

            #y[i,...] = to_categorical(mask, num_classes = 2)

             
        return x, y


    def visualize_batch(self, batch_index = -1):

            N_batches = self.__len__()

            if batch_index >= N_batches:
                print("indice batch oltre il limite di {}".format(N_batches))
                return

            if batch_index == -1:
                batch_index = int(np.random.randint(0, N_batches, 1))

            batch_img, batch_mask = self.__getitem__(batch_index)

            print("batch_img.shape: {}".format(batch_img.shape))
            print("batch_mask.shape: {}".format(batch_mask.shape))

            for i in range(0,len(batch_img)):

                N_subplots = 1 + batch_mask.shape[-1]
                f,arraxis = plt.subplots(1,N_subplots)
                arraxis[0].imshow(batch_img[i])
                #print("N_subplots: {}".format(N_subplots))
                for j in range(N_subplots-1):
                    arraxis[j+1].imshow(batch_mask[i,:,:,j])
                plt.show()

                #histogram_mask, bin_edges_mask = np.histogram(batch_mask[i,:,:,0], bins=256, range=(0, 255))
                #nonz = np.array(np.nonzero(histogram))
                nonz = np.unique(batch_mask[i])
                print("non zero element of hist: {}".format(len(nonz)))
                print(nonz)
                print("min/max img vals: {}/{} ".format(np.min(batch_img[i]), np.max(batch_img[i])))
                print("min/max mask vals: {}/{} ".format(np.min(batch_mask[i,:,:,:]), np.max(batch_mask[i,:,:,:])))