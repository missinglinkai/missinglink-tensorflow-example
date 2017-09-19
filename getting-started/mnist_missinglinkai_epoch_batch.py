# Adopted and modified from TensorFlow's MNIST example.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
#
# In this example, we will build a simple neural network with 2 fully connected layers.
# We will then integrate MissingLink SDK in order to remotely monitor our training, validation
# and testing process.

import os
import math
import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import missinglink

# Input params
NUM_CLASSES = 10  # The MNIST dataset has 10 classes, representing the digits 0 through 9.
IMAGE_SIZE = 28  # The MNIST images are always 28x28 pixels.
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Network params
HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 32

# Training params
LEARNING_RATE = 0.01
MAX_STEPS = 2000
BATCH_SIZE = 100

# Directory for saving input
INPUT_DATA_DIR = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/input_data')

# MissingLink credential
OWNER_ID = "Fill in your owner id"
PROJECT_TOKEN = "Fill in your project token"


def inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
    """
    # Hidden 1 layer
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2 layer
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Linear layer
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
    """Fills the feed_dict using the provided data_set
    """
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
    return {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed,
    }


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    return precision


def run_training():
    """Train MNIST for a number of steps."""
    data_sets = input_data.read_data_sets(INPUT_DATA_DIR)

    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))

        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, HIDDEN1_UNITS, HIDDEN2_UNITS)

        # Add to the Graph the Ops for loss calculation.
        labels = tf.to_int64(labels_placeholder)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        # Add to the Graph the Ops that calculate and apply gradients.
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        # Initialize the graph
        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        # Now that our neural net is ready, let's integrate MissingLinkAI SDK and start the training!

        # Create a project manager with credentials to communicate with MissingLinkAI's backend
        missinglink_project = missinglink.TensorFlowProject(OWNER_ID, PROJECT_TOKEN)

        # Create an experiment as a context manager so MissingLinkAI can monitor the
        # progress of the experiment.
        with missinglink_project.create_experiment(
                display_name='MNIST multilayer perception',
                description='Two fully connected hidden layers') as experiment:

            NUM_SAMPLE = 2000
            NUM_BATCHES = int(NUM_SAMPLE / BATCH_SIZE)

            for epoch in experiment.epoch_loop(10):
                for batch in experiment.batch_loop(NUM_BATCHES):
                    feed_dict = fill_feed_dict(data_sets.train,
                                               images_placeholder, labels_placeholder)

                    # Use `experiment.train` scope before the `session.run` which runs the optimizer
                    # to let the SDK know it should collect the metrics as training metrics.
                    with experiment.train(
                        monitored_metrics={'loss': loss, 'acc': eval_correct}):
                        # Note that you only need to provide the optimizer op. The SDK will automatically run the metric
                        # tensors provided in the `experiment.train` context (and `experiment` context).
                        _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

                # Validate the model with the validation dataset
                with experiment.validation(
                    monitored_metrics={'loss': loss, 'acc': eval_correct}):
                    do_eval(session, eval_correct, images_placeholder,
                            labels_placeholder, data_sets.validation)

            # Use `experiment.test` generator to manage the testing loop.
            total_test_iterations = data_set.num_examples

            with experiment.test(
                total_test_iterations,
                expected=labels_placeholder,
                predicted=logits):
                sess.run([train_op, loss], feed_dict=feed_dict)

if __name__ == '__main__':
    # Provide an alternative to provide MissingLinkAI credential
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-id')
    parser.add_argument('--project-token')

    # Override credential values if provided as arguments
    args = parser.parse_args()
    OWNER_ID = args.owner_id or OWNER_ID
    PROJECT_TOKEN = args.project_token or PROJECT_TOKEN

    run_training()
