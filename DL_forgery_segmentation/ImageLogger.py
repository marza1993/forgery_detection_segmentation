import keras
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from data_generator import data_loader
import matplotlib.pyplot as plt


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image



def image_grid_predictions_gt(N_imgs, batch_img, batch_mask, predictions):
     
    f,arraxis = plt.subplots(N_imgs,3)
    arraxis[0, 0].set_title('image')
    arraxis[0, 1].set_title('groundtruth')
    arraxis[0, 2].set_title('prediction')

    for i in range(0,N_imgs):
        arraxis[i, 0].imshow(batch_img[i])
        arraxis[i, 1].imshow(batch_mask[i,...,0])
        arraxis[i, 2].imshow(predictions[i,...,0])

    return f



class ImageHistory(keras.callbacks.Callback):


    def __init__(self, tensorboard_dir, data_gen, draw_interval = 1, N_imgs = 3):
        super().__init__()

        self.tensor_board_dir = tensorboard_dir
        self.data_gen = data_gen
        self.draw_interval = draw_interval
        self.N_imgs = N_imgs
        # TODO controllo con N_imgs e dimensione del batch


    def on_epoch_end(self, epoch, logs=None):

        # ottengo il primo batch dal data generator
        batch_img, batch_mask = self.data_gen.__getitem__(0)

        # effettuo la predizione sul primo batch
        predictions = self.model.predict(batch_img, batch_img.shape[0])

        # costruico la griglia con img|GT|predizione, per ogni immagine
        figure_grid = image_grid_predictions_gt(self.N_imgs, batch_img, batch_mask, predictions)

        # scrivo le immagini nel formato di tensorboard
        file_writer = tf.summary.create_file_writer(self.tensor_board_dir)
        with file_writer.as_default():
            tf.summary.image("Training evolution", plot_to_image(figure_grid), step=epoch)


  
    #def on_batch_end(self, batch, logs={}):
    #    if batch % self.draw_interval == 0:

    #        # ottengo il primo batch dal data generator
    #        batch_img, batch_mask = self.data_gen.__getitem__(0)

    #        # effettuo la predizione sul primo batch
    #        predictions = self.model.predict(batch_img, batch_img.shape[0])

    #        # costruico la griglia con img|GT|predizione, per ogni immagine
    #        figure_grid = image_grid_predictions_gt(self.N_imgs, batch_img, batch_mask, predictions)

    #        # scrivo le immagini nel formato di tensorboard
    #        file_writer = tf.summary.create_file_writer(self.tensor_board_dir)
    #        with file_writer.as_default():
    #            tf.summary.image("Training evolution per batch", plot_to_image(figure_grid), step=batch)





    








