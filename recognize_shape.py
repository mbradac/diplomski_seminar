import tensorflow as tf
import glob
import os
import argparse
from recognize_common import parse_label, input_fn

tf.logging.set_verbosity(tf.logging.INFO)


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
        optimizer = tf.train.AdamOptimizer()
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
                      default=1000, type=int)
    argp.add_argument("--nocrop", help="If set images will not be croped",
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
            args.model_type + ("__crop" if not args.nocrop else "")
    model_dir = os.path.join("/tmp", model_name)
    model_fn = globals()[args.model_type]
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    if args.max_steps:
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
                train_image_paths, train_label_paths,
                True, True, not args.nocrop, 100),
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
