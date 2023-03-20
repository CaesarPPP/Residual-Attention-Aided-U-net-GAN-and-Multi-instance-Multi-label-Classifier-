import scipy

from keras.datasets import mnist
#from keras_contrib.layers.normalization import InstanceNormalization
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,add,concatenate,Multiply,MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import load_dataset
import numpy as np
import os

import keras.backend as K


class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'overlap'
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                              img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        print(self.discriminator.summary())
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=False):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,
                       padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            #u = BatchNormalization(momentum=0.8)(u)
            #u = Concatenate()([u, skip_input])# skip connection of Unet
            return u
        def attention(layer_input, skip_input):

            au1=Conv2D(1,1,strides=1,padding='same')(layer_input)
            au2=Conv2D(1,1,strides=1,padding='same')(skip_input)
            au1_up=UpSampling2D(size=2)(au1)
            au3=add([au1_up,au2])
            au4=Activation('relu')(au3)
            au5=Conv2D(1,1,strides=1,padding='same')(au4)
            au6=Activation('sigmoid')(au5)
            au7=Multiply()([au6,skip_input])
            return au7
        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)

        # Upsampling
        au1 = attention(d6,d5)
        u1 = deconv2d(d6, d5, self.gf * 8)
        l_con1 = concatenate([au1,u1],axis=3)

        au2 = attention(u1,d4)
        u2 = deconv2d(l_con1, d4, self.gf * 8)
        l_con2 = concatenate([au2,u2],axis=3)

        au3 = attention(u2,d3)
        u3 = deconv2d(l_con2, d3, self.gf * 4)
        l_con3 = concatenate([au3,u3],axis=3)

        au4 = attention(u3,d2)
        u4 = deconv2d(l_con3, d2, self.gf * 2)
        l_con4 = concatenate([au4,u4],axis=3)

        au5 = attention(u4,d1)
        u5 = deconv2d(l_con4, d1, self.gf)
        l_con5 = concatenate([au5,u5],axis=3)

        u6 = UpSampling2D(size=2)(l_con5)

        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same', activation='tanh')(u6)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        imgs_A, imgs_B = dataset_own_keras.input_data_train()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            n = np.random.randint(0, 80)
            hr = imgs_A[n, :, :, :]
            lr = imgs_B[n, :, :, :]
            hr = hr.reshape(1, 128, 128, 3)
            lr = lr.reshape(1, 128, 128, 3)
            # for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(lr)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(
                [hr, lr], valid)
            d_loss_fake = self.discriminator.train_on_batch(
                [fake_A, lr], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            p = np.random.randint(0, 80)
            img_hr = imgs_A[p, :, :, :]
            img_lr = imgs_B[p, :, :, :]
            img_hr = img_hr.reshape(1, 128, 128, 3)
            img_lr = img_lr.reshape(1, 128, 128, 3)
            g_loss = self.combined.train_on_batch(
                [img_hr, img_lr], [valid, img_hr])

            elapsed_time = datetime.datetime.now() - start_time
            if epoch % sample_interval == 0:
                # Plot the progress
                print("[D loss: %f, acc: %3d%%] [G loss: %f] time: %s" %
                      (d_loss[0], 100 * d_loss[1], g_loss[0], elapsed_time))

            # If at save interval => save generated image samples

                self.sample_images(epoch)

    def sample_images(self, epoch):
        os.makedirs('images_overlap/%s' % self.dataset_name, exist_ok=True)
        os.makedirs('g_model_overlap', exist_ok=True)
        r, c = 1, 3

        imgs_A, imgs_B = dataset_own_keras.input_data_train()
        p = np.random.randint(0, 80)
        imgs_hr = imgs_A[p, :, :, :]
        imgs_lr = imgs_B[p, :, :, :]
        imgs_hr = imgs_hr.reshape(1, 128, 128, 3)
        imgs_lr = imgs_lr.reshape(1, 128, 128, 3)
        fake_A = self.generator.predict(imgs_lr)

        gen_imgs = np.concatenate([imgs_lr, fake_A, imgs_hr])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        filename1 = 'g_model_overlap_%04d.h5' % (epoch)
        path = 'E:/test/g_model_overlap/'
        self.generator.save(path + filename1)

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        # for i in range(r):
        for j in range(c):
            # print(gen_imgs[cnt].shape)
            axs[j].imshow(gen_imgs[cnt].reshape(
                256, 256, 3), interpolation='nearest')
            axs[j].set_title(titles[j])
            axs[j].axis('off')
            cnt += 1
        fig.savefig("images_overlap/%s/%d.png" %
                    (self.dataset_name, epoch))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=30000, batch_size=1, sample_interval=80)
    gan.build_generator().summary()
