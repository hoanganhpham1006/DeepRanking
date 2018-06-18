import argparse
import os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding

class get_embedding():
    def __init__(self, model):
        self.model = model

    def _convnet_model_(self):
        vgg_model = VGG16(weights=None, include_top=False)
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
        convnet_model = Model(inputs=vgg_model.input, outputs=x)
        return convnet_model

    def _deep_rank_model(self):
        convnet_model = self._convnet_model_()
        first_input = Input(shape=(228,228,3))
        first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
        first_max = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='same')(first_conv)
        first_max = Flatten()(first_max)
        first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

        second_input = Input(shape=(228,228,3))
        second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
        second_max = MaxPool2D(pool_size=(7,7),strides = (4,4),padding='same')(second_conv)
        second_max = Flatten()(second_max)
        second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

        merge_one = concatenate([first_max, second_max])

        merge_two = concatenate([merge_one, convnet_model.output])
        emb = Dense(4096)(merge_two)
        l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

        final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

        return final_model

    def predict_embedding(self, image):
        model = self._deep_rank_model()
        model.load_weights(self.model)

        image = load_img(image)
        image = img_to_array(image).astype("float64")
        image = transform.resize(image, (228, 228))
        image *= 1. / 255
        image = np.expand_dims(image, axis = 0)

        embedding = model.predict([image, image, image])[0]
        return embedding

# ge = get_embedding("./deepranking.h5")
# print(ge.predict_embedding("./1.jpg").shape)
