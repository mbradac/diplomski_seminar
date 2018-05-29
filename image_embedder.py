import tensorflow as tf
import glob
import os
import argparse
import numpy as np
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

def input_fn(image_paths, label_paths, repeat=False, shuffle=False,
             crop=False, flip=False, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

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
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def image_embedder(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[4, 4],
            strides=[4, 4])

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=[2, 2])

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 32])
    dense = tf.layers.dense(
            inputs=pool2_flat, units=512, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    logits_color = tf.layers.dense(inputs=dropout, units=3)
    logits_shape = tf.layers.dense(inputs=dropout, units=3)

    color_class = tf.argmax(input=logits_color, axis=1)
    shape_class = tf.argmax(input=logits_shape, axis=1)
    predictions = {
            "classes_color": color_class,
            "classes_shape": shape_class,
            "classes": color_class * 3 + shape_class,
            "probabilities_color":
                    tf.nn.softmax(logits_color, name="softmax_tensor_color"),
            "probabilities_shape":
                    tf.nn.softmax(logits_shape, name="softmax_tensor_shape"),
            "embedding": dense,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels_color = labels["color"]
    labels_shape = labels["shape"]
    combined_labels = labels_color * 3 + labels_shape
    loss_color = tf.losses.sparse_softmax_cross_entropy(
            labels=labels_color, logits=logits_color)
    loss_shape = tf.losses.sparse_softmax_cross_entropy(
            labels=labels_shape, logits=logits_shape)
    loss = loss_color + loss_shape

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            "accuracy_color": tf.metrics.accuracy(
                    labels=labels_color,
                    predictions=predictions["classes_color"]),
            "accuracy_shape": tf.metrics.accuracy(
                    labels=labels_shape,
                    predictions=predictions["classes_shape"]),
            "accuracy": tf.metrics.accuracy(
                    labels=combined_labels,
                    predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    argp = argparse.ArgumentParser()
    argp.add_argument("--embed", help="Only embed, don't train",
                      action="store_true")
    argp.add_argument("--max_steps", help="Max steps to train the model",
                      default=1000, type=int)
    argp.add_argument("--nocrop", help="If set images will not be croped",
                      action="store_true")
    argp.add_argument("input", help="Directory with images and labels")
    args = argp.parse_args()

    image_paths = glob.glob(os.path.join(args.input, "pic*"))
    label_paths = list(map(lambda path:
                           path.replace("pic", "shape").replace(".png", ""),
                           image_paths))

    train_size = int(len(image_paths) * 0.7)
    train_image_paths = image_paths[:train_size]
    test_image_paths = image_paths[train_size:]
    train_label_paths = label_paths[:train_size]
    test_label_paths = label_paths[train_size:]

    model_name = "image_embedder"
    model_dir = os.path.join("/tmp", model_name)
    model_fn = image_embedder
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    tensors_to_log = {"probabilities_color": "softmax_tensor_color",
                      "probabilities_shape": "softmax_tensor_shape"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    if args.max_steps and not args.embed:
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
                train_image_paths, train_label_paths,
                True, True, not args.nocrop, 100),
                max_steps=args.max_steps, hooks=[logging_hook])

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
                test_image_paths, test_label_paths))

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    predict_results = classifier.predict(input_fn=lambda: input_fn(
        image_paths, label_paths, crop=True, flip=True))
    for i, prediction in enumerate(predict_results):
        embedding_filename = "embedding{:06}".format(i)
        embedding_filepath = os.path.join(args.input, embedding_filename)
        embedding_label_filepath = embedding_filepath.replace("embedding", "embedding_label")
        np.save(embedding_filepath, prediction["embedding"])
        shutil.copyfile(label_paths[int(i / 10)], embedding_label_filepath)

    print(list(predict_results))


if __name__ == "__main__":
    tf.app.run()
