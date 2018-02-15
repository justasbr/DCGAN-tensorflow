from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from skimage.color import rgb2gray

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=42, sample_num=42, output_height=64, output_width=64,
                 z_dim=100, gf_dim=64, df_dim=64, y_dim=None,
                 c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1

        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_model(self):

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.input_rgb = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        self.input_grayscale = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 1], name='input_grayscale')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.input_grayscale)
        self.D, self.D_logits = self.discriminator(self.input_rgb, self.input_grayscale, reuse=False)
        self.sampler = self.sampler(self.input_grayscale)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.input_grayscale, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.log(self.D_))
        self.g_distance = tf.reduce_mean(tf.abs(self.input_rgb - self.G))
        self.g_loss += self.g_distance
        # tf.reduce_mean(
        # sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_train_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_train_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = self.data[0:self.sample_num]
        sample = [
            get_image(sample_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for sample_file in sample_files]

        sample_inputs_grayscale = [rgb2gray(color_img) for color_img in sample]
        sample_inputs_grayscale = np.array(sample_inputs_grayscale)
        sample_inputs_grayscale = np.expand_dims(sample_inputs_grayscale, axis=3).astype(np.float32)  # (BSIZE,64,64,1)

        if (self.grayscale):
            sample_inputs_rgb = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs_rgb = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Checkpoint not found...")

        for epoch in xrange(config.epoch):
            self.data = glob(os.path.join(
                "./data", config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=self.crop,
                              grayscale=self.grayscale) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)  # (BSIZE, 64,64,3)

                batch_grayscale = [rgb2gray(color_img) for color_img in batch]
                batch_images_grayscale = np.array(batch_grayscale)
                batch_images_grayscale = np.expand_dims(batch_images_grayscale, axis=3).astype(
                    np.float32)  # (BSIZE,64,64,1)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                if np.random.rand() > 0.9:  # mess with discriminator
                    print("Messing")

                    batch_images = self.sess.run(self.sampler, feed_dict={self.input_grayscale: batch_images_grayscale,
                                                                          self.z: batch_z})
                # Update D network
                _, summary_str = self.sess.run([d_train_opt, self.d_sum],
                                               feed_dict={self.input_rgb: batch_images,
                                                          self.z: batch_z,
                                                          self.input_grayscale: batch_images_grayscale})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_train_opt, self.g_sum],
                                               feed_dict={self.z: batch_z,
                                                          self.input_grayscale: batch_images_grayscale,
                                                          self.input_rgb: batch_images})
                self.writer.add_summary(summary_str, counter)

                # Run g_train_opt twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_train_opt, self.g_sum],
                                               feed_dict={self.z: batch_z,
                                                          self.input_grayscale: batch_images_grayscale,
                                                          self.input_rgb: batch_images})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z,
                                                   self.input_grayscale: batch_images_grayscale,
                                                   self.input_rgb: batch_images})
                errD_real = self.d_loss_real.eval(
                    {self.input_rgb: batch_images, self.input_grayscale: batch_images_grayscale})
                errG = self.g_loss.eval({self.z: batch_z,
                                         self.input_grayscale: batch_images_grayscale,
                                         self.input_rgb: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 20) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.input_grayscale: sample_inputs_grayscale,
                                self.z: sample_z,
                                self.input_rgb: sample_inputs_rgb
                            },
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except Exception as e:
                        print("one pic error!...", e)

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, images_rgb, images_grayscale, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            images_concat = tf.concat([images_rgb, images_grayscale], axis=3)  # (B, 64, 64, 4)

            d1 = lrelu(self.d_bn1(conv2d(images_concat, output_dim=64, name="d_conv1")))  # (B, 32, 32, 64)

            d2 = lrelu(self.d_bn2(conv2d(d1, output_dim=128, name="d_conv2")))  # (B, 16, 16, 128)

            d3 = lrelu(self.d_bn3(conv2d(d2, output_dim=256, name="d_conv3")))  # (B, 8, 8, 256)

            # d4 = lrelu(self.d_bn4(conv2d(d3, output_dim=512, name="d_conv4")))  # (B, 4, 4, 512)
            # d5 = lrelu(self.d_bn5(conv2d(d4, output_dim=1024, name="d_conv5")))  # (B, 2, 2, 1024)

            d6 = linear(tf.reshape(d3, [self.batch_size, -1]), 1, 'd_linear')

            return tf.nn.sigmoid(d6), d6

    def generator(self, grayscale, reuse=False):
        with tf.variable_scope("generator", reuse=reuse) as scope:
            down1 = lrelu(conv2d(grayscale, output_dim=64, name='g_conv_down1'))  # (32,32,64)
            down2 = lrelu(conv2d(down1, output_dim=128, name='g_conv_down2'))  # (16,16,128)
            down3 = lrelu(conv2d(down2, output_dim=256, name='g_conv_down3'))  # (8,8,256)
            # down4 = lrelu(conv2d(down3, output_dim=512, name='g_conv_down4'))  # (4,4,512)
            # down5 = lrelu(conv2d(down4, output_dim=1024, name='g_conv_down5'))  # (2,2,1024)

            # up1 = lrelu(
            #     conv2d_transpose(down5, output_shape=[self.batch_size, 4, 4, 512], name="g_conv_up1"))  # (4,4,512)
            # up1 = tf.concat([up1, down4], axis=3)  # 4,4,1024

            # up2 = lrelu(
            #     conv2d_transpose(down4, output_shape=[self.batch_size, 8, 8, 256], name="g_conv_up2"))  # (r8,8,256)
            # up2 = tf.concat([up2, down3], axis=3)  # 8,8,512

            up3 = lrelu(
                conv2d_transpose(down3, output_shape=[self.batch_size, 16, 16, 128], name="g_conv_up3"))  # (16,16,128)
            up3 = tf.concat([up3, down2], axis=3)  # 16,16,256

            up4 = lrelu(conv2d_transpose(up3, output_shape=[self.batch_size, 32, 32, 64], name="g_conv_up4"))
            up4 = tf.concat([up4, down1], axis=3)  # 32,32,128

            up5 = lrelu(conv2d_transpose(up4, output_shape=[self.batch_size, 64, 64, 3], name="g_conv_up5"))
            up5 = tf.concat([up5, grayscale], axis=3)  # 64,64,4

            output_rgb = conv2d(up5, stride_h=1, stride_w=1, output_dim=3, name="g_conv_final")
            return tf.nn.sigmoid(output_rgb)

    def sampler(self, grayscale):
        return self.generator(grayscale, reuse=True)
        # with tf.variable_scope("generator") as scope:
        #     scope.reuse_variables()
        #
        #     down1 = lrelu(conv2d(grayscale, output_dim=64, name='g_conv1'))  # (32,32,64)
        #
        #     down2 = lrelu(conv2d(down1, output_dim=128, name='g_conv2'))  # (16,16,128)
        #
        #     up1 = lrelu(
        #         conv2d_transpose(down2, output_shape=[self.batch_size, 32, 32, 64], name="g_conv3"))  # (32,32, 64)
        #     up1 = tf.concat([up1, down1], axis=3)
        #
        #     up2 = lrelu(conv2d_transpose(up1, output_shape=[self.batch_size, 64, 64, 3], name="g_conv4"))  # (64,64,3)
        #     up2 = tf.concat([up2, grayscale], axis=3)  # 4 channels #(64,64,4)
        #
        #     output_rgb = lrelu(conv2d(up2, stride_h=1, stride_w=1, output_dim=3, name="g_conv5"))
        #     return output_rgb

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
