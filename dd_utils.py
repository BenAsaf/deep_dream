"""
Made by Ben Asaf

    ToBenAsaf@Gmail.com
    https://github.com/BenAsaf
    https://www.linkedin.com/in/ben-asaf/

    Oct 10 2020
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple, Union, List


def save_image(image: np.array, path: str):
    with open(path, 'wb') as file:
        Image.fromarray(image).save(file, 'jpeg')
    print(f"Saved image at: {path}")


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8)])
def process_image(image: tf.Tensor):
    image = tf.cast(image, dtype=tf.float32)
    out = image[:, :, ::-1]  # RGB -> BGR
    out *= 1/255.0  # [0,255]->[0,1]
    out *= 2.0  # [0,1]->[0,2]
    out = (out-1.0)  # [0,2]->[-1,1]
    out = tf.expand_dims(out, 0)
    return out


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
def deprocess_image(image: tf.Tensor):
    out = tf.squeeze(image)
    out = out[:, :, ::-1]
    out += 1.0  # [-1,1]->[0,2]
    out /= 2.0  # [0,2]->[0,1]
    out *= 255.0  # [0,1]->[0,255]
    out = tf.clip_by_value(out, 0.0, 256.0)
    return tf.cast(out, dtype=np.uint8)


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
                     tf.TensorSpec(shape=[], dtype=tf.int32),
                     tf.TensorSpec(shape=[], dtype=tf.float32))
)
def gaussian_blur(image: tf.Tensor, kernel_size: int = 9, sigma: float = 0.5):

    def gauss_kernel(num_channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, num_channels])
        return kernel

    kernel_size = tf.cast(kernel_size, tf.float32)
    gaussian_kernel = gauss_kernel(num_channels=tf.shape(image)[-1], kernel_size=kernel_size, sigma=sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]
    result = tf.nn.depthwise_conv2d(image, gaussian_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')
    return result


# @tf.function
def resize_image(image: tf.Tensor, factor: Union[float, List[float]] = None, size: Tuple[int, int] = None,
                 resize_method: tf.image.ResizeMethod = tf.image.ResizeMethod.LANCZOS3, antialias: bool = True):
    if size is not None:
        size = size  # just so it's clear
    elif factor is not None:
        _shape = tf.multiply(factor, tf.cast(tf.shape(image)[1:3], dtype=tf.float32))
        size = tf.cast(_shape, dtype=tf.int32)
    else:
        raise ValueError("Need either 'factor' or 'size'")
    image = tf.image.resize(images=image, size=size, method=resize_method, antialias=antialias)
    return image


@tf.function
def clip(image):
    image = tf.clip_by_value(image, -1, 1)
    return image
