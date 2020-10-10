"""
Made by Ben Asaf

    ToBenAsaf@Gmail.com
    https://github.com/BenAsaf
    https://www.linkedin.com/in/ben-asaf/

    Oct 10 2020
"""
import tensorflow as tf
import numpy as np
from typing import List
from PIL import Image
import argparse
import os
from tqdm.auto import tqdm

import dd_utils

tf.get_logger().setLevel("ERROR")


class DeepDream(tf.Module):

    def __init__(self, layer_names: List[str], base_model: tf.keras.Model):
        super().__init__()
        self._base_model = base_model
        output_layers = [self._base_model.get_layer(n).output for n in layer_names]
        self.model = tf.keras.Model(inputs=self._base_model.inputs, outputs=output_layers)

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, image: tf.Tensor, iterations: int, lr: float):
        """ Updates the image to maximize outputs for n iterations """
        iterations_float = tf.cast(iterations, dtype=tf.float32)
        for n in tf.range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(image)
                out = self.model(image)
                loss = tf.constant(0.0, dtype=tf.float32)
                if len(out) == 1:
                    loss += tf.norm(out, ord="euclidean")
                else:
                    for x in out:
                        loss += tf.norm(x, ord="euclidean")
            grads = tape.gradient(loss, image)

            sigma = (tf.cast(n, tf.float32) * 2.0) / iterations_float + 0.5
            grad_smooth1 = dd_utils.gaussian_blur(grads, kernel_size=9, sigma=sigma)
            grad_smooth2 = dd_utils.gaussian_blur(grads, kernel_size=9, sigma=sigma * 2)
            grad_smooth3 = dd_utils.gaussian_blur(grads, kernel_size=9, sigma=sigma * 0.5)

            grads = (grad_smooth1 + grad_smooth2 + grad_smooth3)

            avg_g = tf.reduce_mean(tf.abs(grads))
            grads /= avg_g
            image += lr * grads
            image = dd_utils.clip(image)
        return image


def deep_dream(image: np.array,
               deepdream: DeepDream,
               iterations: int,
               lr: float,
               octave_scale: float,
               num_octaves: int):
    """ Main deep dream method """
    image = tf.constant(image, dtype=tf.uint8)
    image = dd_utils.process_image(image=image)

    antialias = False
    resize_method = tf.image.ResizeMethod.LANCZOS5

    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(
            dd_utils.resize_image(image=dd_utils.gaussian_blur(octaves[-1], sigma=0.5),  # Blur before downsampling
                                  factor=octave_scale,
                                  resize_method=resize_method,
                                  antialias=antialias)
        )

    dreamed_image = image
    details = tf.zeros_like(octaves[-1])
    for octave_idx, octave_base in enumerate(tqdm(octaves[::-1])):
        if octave_idx > 0:
            details = dd_utils.resize_image(image=details, size=octave_base.shape[1:3], resize_method=resize_method,
                                            antialias=antialias)
        input_image = octave_base + details  # Add the Details
        dreamed_image = deepdream(input_image, iterations, lr)  # Hallucinate moreeeee
        details = dreamed_image - octave_base  # Subtract to get the delta to the image, the Details
    output_image = dd_utils.deprocess_image(dreamed_image)
    return output_image.numpy()


def parse_args() -> argparse.Namespace:
    _default_output_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to an input image")
    parser.add_argument("--output_dir", type=str, default=_default_output_dir,
                        help="Path to a directory where the output will be saved.\n"
                             "Default: Same directory in 'outputs' dir.")
    parser.add_argument("--layer_names", default=["mixed3", "mixed5"], nargs="+",
                        help="Layer at which we modify image to maximize outputs.\n"
                             "Choose either: 'mixed0', 'mixed1', ...,'mixed10' or a combination of them.\n"
                             "Default: ['mixed3', 'mixed5']")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="Learning rate.\n"
                             "Default: 0.01")
    parser.add_argument("--octave_scale", default=0.75, type=float,
                        help="Image scale between octaves.\n"
                             "Default: 0.75")
    parser.add_argument("--num_octaves", default=None, type=int,
                        help="Number of octaves; How many downsampling to do and apply DeepDream.\n"
                             "Default determines the maximal num octaves possible.")
    parser.add_argument("--iterations", default=10, type=int,
                        help="Number of Gradient Ascent steps per octave.\n"
                             "Default: 10")
    args = parser.parse_args()

    if not os.path.exists(args.input_image):  # Make sure it exists.
        raise ValueError(f"Sorry! 'input_image' does not exist at: {args.input_image}")

    args.output_dir = os.path.expanduser(args.output_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, "outputs")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def calculate_maximal_num_octaves(image: np.array, octave_scale: float, minimal_height_width: int):
    shape = np.array(image.shape[:2], dtype=np.float32)  # H, W
    maximal_octave = 0
    while np.all(np.floor(shape) > minimal_height_width):
        shape *= octave_scale
        maximal_octave += 1
    return maximal_octave


def main():
    args = parse_args()

    image = np.array(Image.open(args.input_image), dtype=np.uint8)  # Load image

    base_model = tf.keras.applications.InceptionV3(include_top=False)  # Our base model
    minimal_height_width = 85  # for InceptionV3 the minimal size for input is (85,85)
    base_model.summary()
    maximal_octave = calculate_maximal_num_octaves(image=image,
                                                   octave_scale=args.octave_scale,
                                                   minimal_height_width=minimal_height_width)
    if args.num_octaves is None:
        args.num_octaves = maximal_octave
    elif args.num_octaves > maximal_octave:
        print(f"Your chosen octave is too high for this model. Setting it to: {maximal_octave}")
        args.num_octaves = maximal_octave

    deepdream = DeepDream(layer_names=args.layer_names,
                          base_model=base_model)

    dreamed_image = deep_dream(
        image=image,
        deepdream=deepdream,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    # Save and plot image
    filename, ext = os.path.splitext(args.input_image.split("/")[-1])
    dd_utils.save_image(image=dreamed_image, path=os.path.join(args.output_dir, f"{filename}_output{ext}"))


if __name__ == '__main__':
    main()
