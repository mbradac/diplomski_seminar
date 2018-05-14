import tensorflow as tf
import glob
import os
import numpy as np
import argparse

tf.logging.set_verbosity(tf.logging.INFO)


def parse_label(label_path):
    with open(label_path) as label_file:
        file_string = label_file.read().strip()
        label_color_string, label_shape_string = file_string.split()
        return (["red", "green", "blue"].index(label_color_string),
                ["triangle", "rectangle", "circle"].index(
                    label_shape_string))


def input_fn(image_paths, labels, repeat=False, shuffle=False,
             crop=False, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

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

    dataset = dataset.map(prepare_label)
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

    if crop:
        dataset = dataset.flat_map(crop_image)
        dataset = dataset.map(lambda x, y: ({"x" : x}, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def _cnn(features, labels, mode, conv1_filters, conv2_filters, dense_units):
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=conv1_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[4, 4],
            strides=[4, 4])

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=conv2_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=[2, 2])

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * conv2_filters])
    dense = tf.layers.dense(
            inputs=pool2_flat, units=dense_units, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = labels["shape"]
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn1(features, labels, mode):
    return _cnn(features, labels, mode, 16, 32, 512)


def cnn2(features, labels, mode):
    return _cnn(features, labels, mode, 32, 64, 1024)


def main(unused_argv):
    argp = argparse.ArgumentParser()
    argp.add_argument("--max_steps", help="Max steps to train the model",
                      default=20000, type=int)
    argp.add_argument("--crop", help="Type of the model",
                      action="store_true")
    argp.add_argument("input", help="Directory with images and labels")
    argp.add_argument("model_type", help="Type of the model",
                      choices=["cnn1", "cnn2"])
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

    model_name = os.path.split(os.path.normpath(args.input))[1] + "__" + \
            args.model_type + ("__crop" if args.crop else "")
    model_dir = os.path.join("/tmp", model_name)
    model_fn = globals()[args.model_type]
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    if args.max_steps:
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
                train_image_paths, train_label_paths,
                True, True, args.crop, 100),
                max_steps=args.max_steps, hooks=[logging_hook])

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
                test_image_paths, test_label_paths))

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    predict_results = classifier.predict(input_fn=lambda: input_fn(
                    test_image_paths, test_label_paths))
    print(list(map(lambda x: (test_image_paths[x[0]], x),
          filter(lambda x: parse_label(test_label_paths[x[0]])[1] \
                  != x[1]["classes"],
          enumerate(predict_results)))))


if __name__ == "__main__":
    tf.app.run()
