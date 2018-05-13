import tensorflow as tf
import glob
import os
import numpy as np
import argparse

tf.logging.set_verbosity(tf.logging.INFO)


def input_fn(image_paths, labels, repeat=False, shuffle=False, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    def parse_function(image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_floats = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_reshaped = tf.reshape(image_floats, [64, 64, 3])
        return {"x": image_reshaped}, label
    dataset = dataset.map(parse_function)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def cnn1(features, labels, mode):
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

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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


def cnn2(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[4, 4],
            strides=[4, 4])

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=[2, 2])

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(
            inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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


def main(unused_argv):
    argp = argparse.ArgumentParser()
    argp.add_argument("--max_steps", help="Max steps to train the model",
                      default=20000, type=int)
    argp.add_argument("input", help="Directory with images and labels")
    argp.add_argument("model_type", help="Type of the model",
                      choices=["cnn1", "cnn2"])
    args = argp.parse_args()

    image_paths = glob.glob(os.path.join(args.input, "pic*"))
    label_paths = list(map(lambda path:
                           path.replace("pic", "shape").replace(".png", ""),
                           image_paths))
    def label_path_to_shape_id(label_path):
        with open(label_path) as label_file:
            label_string = label_file.read().strip()
            return ["triangle", "rectangle", "circle"].index(label_string)
    labels = np.asarray(list(map(label_path_to_shape_id, label_paths)),
                        dtype=np.int32)
    train_size = int(len(image_paths) * 0.7)
    train_image_paths = image_paths[:train_size]
    test_image_paths = image_paths[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    model_name = os.path.split(os.path.normpath(args.input))[1] + "__" + \
            args.model_type
    model_dir = os.path.join("/tmp", model_name)
    model_fn = globals()[args.model_type]
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    if args.max_steps:
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
                train_image_paths, train_labels, True, True, 100),
                max_steps=args.max_steps, hooks=[logging_hook])

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
                test_image_paths, test_labels))

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    predict_results = classifier.predict(input_fn=lambda: input_fn(
                    test_image_paths, test_labels))
    print(list(map(lambda x: (test_image_paths[x[0]], x),
          filter(lambda x: test_labels[x[0]] != x[1]["classes"],
          enumerate(predict_results)))))


if __name__ == "__main__":
    tf.app.run()
