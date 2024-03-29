import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, SeparableConv2D
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from keras.layers import concatenate, add
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras import Model



class segmentation_model(object):

    """
    classe che rappresenta il modello per effettuare la classificazione binaria FORGED/ORIGINAL
    """


    #def build_unet(pretrained_weights = None,input_size = (512,512,3)):

    #    inputs = Input(input_size)
    #    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    #    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    #    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    #    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #    drop4 = Dropout(0.5)(conv4)
    #    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    #    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #    drop5 = Dropout(0.5)(conv5)

    #    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #    merge6 = concatenate([drop4,up6], axis = 3)
    #    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    #    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #    merge7 = concatenate([conv3,up7], axis = 3)
    #    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    #    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #    merge8 = concatenate([conv2,up8], axis = 3)
    #    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    #    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #    merge9 = concatenate([conv1,up9], axis = 3)
    #    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    #    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    #    conv10 = Conv2D(1, 3, activation = 'sigmoid', padding = 'same')(conv9)

    #    outputs = conv10

    #    #outputs = Reshape()

    #    #conv10 = Conv2D(2, 3, padding = 'same')(conv9)

    #    #outputs = tfa.layers.Sparsemax(-1)(conv10)


    #    #model = Model(input = inputs, output = conv10)

    #    model = Model(input = inputs, output = outputs)

    #    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    
    #    #model.summary()

    #    if (pretrained_weights):
    #        model.load_weights(pretrained_weights)
    #    return model




    def build_unet(pretrained_weights = None,input_size = (512,512,3), num_classes = 2):

        inputs = Input(input_size)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPool2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        previous_block_activation = x  # Set aside residual

        for filters in [256, 128, 64, 32]:
            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = UpSampling2D(2)(x)

            # Project residual
            residual = UpSampling2D(2)(previous_block_activation)
            residual = Conv2D(filters, 1, padding="same")(residual)
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual


        # Add a per-pixel classification layer
        outputs = Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        # Define the model
        model = Model(inputs, outputs)

        if pretrained_weights:
            model.load_weights(pretrained_weights)
        return model








