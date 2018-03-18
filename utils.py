"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
from glob import glob
from skimage.color import rgb2gray

import scipy.misc
import numpy as np
import os
from scipy.misc.pilutil import bytescale

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(img, crop_h, crop_w,
                resize_h=64, resize_w=64):
    y, x = img.shape[:2]
    startx = x // 2 - 32
    starty = y // 2 - 32
    return img[starty:starty + 64, startx:startx + 64]


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def validate(sess, dcgan, config, epoch=None, sampling=True, write_loss=True, val_data=None):
    dataset = config.dataset if val_data is None else val_data
    data_folder = os.path.join("./data", dataset, "val", config.input_fname_pattern)
    image_files = glob(data_folder)
    num_batches = int( len(image_files) / config.batch_size)
    print("NumBatches", num_batches) 
    data_len = num_batches * config.batch_size
    print("Data len", data_len) 
    data = load_images(config, image_files[:data_len])

    total_L1 = 0.0
    total_L2 = 0.0

    for batch_id in range(num_batches):
        batch = get_data_batch(batch_id, config, data)
        batch_rgb = make_rgb(batch)
        batch_grayscale = make_grayscale(batch)

        assert batch_rgb.shape == (config.batch_size, 64, 64, 3)
        assert batch_grayscale.shape == (config.batch_size, 64, 64, 1)

        samples = get_samples(batch_grayscale, dcgan, sess)
        if sampling:
            save_samples(batch_id, batch_rgb, config, dataset.split("/")[-1], epoch, samples)

        L1, L2 = calc_losses(batch_rgb, samples)
        total_L1 += L1
        total_L2 += L2

    loss_data = "%s\t%s\t%s\t%.2f\t%.2f\n" % (
        dataset, config.train_size, epoch, total_L1 / data_len, total_L2 / data_len)
    print("Data\tTsize\tepoch\tL1\tL2")
    print(loss_data)
    if write_loss:
        with open("losses.out", "a") as f:
            f.write(loss_data)


def save_samples(batch_id, batch_rgb, config, dataset, epoch, samples):
    if batch_id == 0:
        real_path = "val_data/val_" + dataset + "_" + str(config.train_size) + "_real_" + str(epoch) + ".png"
        gen_path = "val_data/val_" + dataset + "_" + str(config.train_size) + "_gen_" + str(epoch) + ".png"
        imsave(batch_rgb, (10, 10), real_path)
        imsave(samples, (10, 10), gen_path)


def get_data_batch(batch_id, config, data):
    start_id = batch_id * config.batch_size
    end_id = start_id + config.batch_size
    batch = data[start_id:end_id]
    return batch


def calc_losses(batch_rgb, samples):
    diff = batch_rgb - samples
    L1 = np.abs(diff)
    L1 = L1.sum(axis=3)
    L1 = L1.mean(axis=(1, 2))  # (batch_size, 1)
    L2 = np.square(diff, dtype=np.float64)
    L2 = L2.sum(axis=3)
    L2 = L2.mean(axis=(1, 2))  # (batch_size, 1)
    return L1.sum(), L2.sum()


def load_images(config, image_files):
    return [get_image(file_name,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=config.output_height,
                      resize_width=config.output_width,
                      crop=config.crop,
                      grayscale=False) for file_name in image_files]


def get_samples(batch_grayscale, dcgan, sess):
    samples = sess.run(
        dcgan.sampler,
        feed_dict={dcgan.input_grayscale: batch_grayscale}
    )
    samples[samples < -1] = -1
    samples[samples > 1] = 1
    samples = inverse_transform(samples)
    samples *= 255
    return samples


def make_rgb(batch):
    batch_rgb = np.array(batch).astype(np.float32)  # (BSIZE, 64,64,3)

    batch_rgb = inverse_transform(batch_rgb)
    batch_rgb *= 255
    batch_rgb = batch_rgb.astype(np.uint8)

    return batch_rgb


def make_grayscale(batch):
    batch_grayscale = [rgb2gray(color_img) for color_img in batch]
    batch_grayscale = np.array(batch_grayscale)
    batch_grayscale = np.expand_dims(batch_grayscale, axis=3).astype(np.float32)  # (BSIZE,64,64,1)
    return batch_grayscale


def validate_labels(sess, dcgan, config, main_folder):
    label_dirs = glob(os.path.join("./data", main_folder, "*"))
    print(label_dirs)
    label_dirs = list(filter(lambda n: os.path.isdir(n), label_dirs))
    print(label_dirs)
    labels = [label_dir.split("/")[-1] for label_dir in label_dirs]
    print(labels)
    for label in labels:
        print("Validating label: ", label)
        label_folder = main_folder + "/" + label
        validate(sess, dcgan, config, sampling=True, write_loss=False, val_data=label_folder)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w
