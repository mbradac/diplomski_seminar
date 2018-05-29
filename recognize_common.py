import tensorflow as tf
import numpy as np


def parse_label(label_path):
    with open(label_path) as label_file:
        file_string = label_file.read().strip()
        label_color_string, label_shape_string = file_string.split()
        return (["red", "green", "blue"].index(label_color_string),
                ["triangle", "rectangle", "circle"].index(
                    label_shape_string))


def input_fn(image_paths, label_paths, repeat=False, shuffle=False,
             crop=False, flip=False, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    def prepare_label(image, label_path):
        color, shape = tf.py_func(parse_label, [label_path],
                                  [tf.int64, tf.int64])
        return image, {"color": color, "shape": shape}

    def prepare_image(image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_floats = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_reshaped = tf.reshape(image_floats, [64, 64, 3])
        return {"x": image_reshaped}, label

    dataset = dataset.map(prepare_image)

    def crop_image(x, label):
        cropped = tf.image.crop_and_resize(
                tf.stack([x["x"]]),
                np.asarray([[0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.9, 0.9],
                            [0.0, 0.1, 0.9, 1.0],
                            [0.1, 0.0, 1.0, 0.9],
                            [0.1, 0.1, 1.0, 1.0]]),
                [0, 0, 0, 0, 0],
                [64, 64])
        return tf.data.Dataset.from_tensor_slices((
                cropped, tf.stack([label] * 5)))

    def flip_image(x, label):
        cropped = tf.image.crop_and_resize(
                tf.stack([x["x"]]),
                np.asarray([[0.0, 0.0, 1.0, 1.0],
                            [0.0, 1.0, 1.0, 0.0]]),
                [0, 0],
                [64, 64])
        return tf.data.Dataset.from_tensor_slices((
                cropped, tf.stack([label] * 2)))

    if crop:
        dataset = dataset.flat_map(crop_image)
        dataset = dataset.map(lambda x, y: ({"x" : x}, y))
    if flip:
        dataset = dataset.flat_map(flip_image)
        dataset = dataset.map(lambda x, y: ({"x" : x}, y))
    dataset = dataset.map(prepare_label)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
