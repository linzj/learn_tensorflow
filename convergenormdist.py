#import numpy as np
#import tensorflow as tf
#import math, random
#from math import e
#
#graph = tf.Graph()
#learning_rate = 1e-3
#hidden_size = 1024 * 3
#
#def build_graph(is_training):
#   r_inputs = tf.placeholder(tf.float32, shape=(1))
#   r_targets = tf.placeholder(tf.float32, shape=(1))
#   _inputs = tf.reshape(r_inputs, shape=(1, 1))
#   _targets = tf.reshape(r_targets, shape=(1, 1))
#
#   w_hidden = tf.get_variable(
#       "w_hidden", [1, hidden_size], dtype=tf.float32)
#   b_hidden = tf.get_variable("b_hidden", [hidden_size], dtype=tf.float32)
#
#   w = tf.get_variable(
#       "w", [hidden_size, 1], dtype=tf.float32)
#   b = tf.get_variable("b", [1], dtype=tf.float32)
#   
#   hidden = tf.matmul(_inputs, w_hidden) + b_hidden
#   logits = tf.matmul(hidden, w) + b
#   #loss = tf.squared_difference(logits, r_targets)
#   loss = tf.losses.mean_squared_error(logits, _targets)
#   sample_op = logits
#   if not is_training:
#       return r_inputs, sample_op
#   train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
#   return train_op, loss, r_inputs, r_targets
#
#def is_correct(a, b):
#    if abs(a[0] - b) < 1.0:
#        return "Correct"
#    else:
#        return "Incorrect"
#
#def main():
#    with graph.as_default():
#        with tf.name_scope("Train"):
#            with tf.variable_scope("Model", reuse=None):
#                train_op, _loss, train_inputs, _targets = build_graph(True)
#
#        with tf.name_scope("Sample"):
#            with tf.variable_scope("Model", reuse=True) as vscope:
#                sample_inputs, sample_op = build_graph(False)
#    n = 0
#    p = 0
#    with tf.Session(graph=graph) as session:
#      # We must initialize all variables before we use them.
#        init = tf.global_variables_initializer()
#        init.run()
#        print("Initialized")
#        train_data = []
#        train_labels = []
#        w = 100.0
#        b = 10.0
#        for i in range(1, 10000):
#            train_data.append(i)
#            train_labels.append(e ** (-1.0 / i) * i)
#        eval_data = []
#        eval_labels = []
#        for i in range(10000, 20000):
#          eval_data.append(i)
#          eval_labels.append(e ** (-1.0 / i) * i)
#        while True:
#            if p >= len(train_data):
#                p = 0
#            _, loss = session.run([train_op, _loss], feed_dict = {train_inputs: train_data[p:p+1],
#                    _targets: train_labels[p:p+1]})
#            if n % 1000 == 0:
#                print 'iter %d, loss: %f' % (n, loss) # print progress
#                start_index = random.randint(0, len(eval_data) - 1)
#                sample_v = session.run([sample_op], feed_dict = {sample_inputs: eval_data[start_index:start_index+1]})
#                print '----\n %f, %f, %s \n----' % (sample_v[0], eval_labels[start_index], is_correct(sample_v, eval_labels[start_index]))
#            n += 1
#            p += 1
#
#if __name__ == '__main__':
#    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
hidden_size = 1024 * 3
learning_rate = 1e-3

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode):
    features = tf.reshape(features, (1, 1), name="features")
    labels = tf.reshape(labels, (1, 1), name="labels")
    
    hidden_layer = tf.contrib.layers.linear(features, hidden_size)
    output_layer = tf.contrib.layers.linear(hidden_layer, 1)

    loss = None
    train_op = None

    prediction = tf.reshape(output_layer, (1, 1), name="prediction")

# Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
      loss = tf.losses.mean_squared_error(prediction, labels)

# Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=learning_rate,
                optimizer="Adagrad")

# Generate Predictions
    predictions = {
      "value": prediction
    }

# Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
  # Load training and eval data

    train_data = []
    train_labels = []
    w = 100.0
    b = 10.0
    f = lambda x :math.exp(-x**2/2)/math.sqrt(2*math.pi)
    for i in range(1, 10000):
        x = i / 10000.0
        train_data.append(x)
        train_labels.append(f(x))
    eval_data = []
    eval_labels = []
    for i in range(10000, 20000):
      x = i / 10000.0
      eval_data.append(x)
      eval_labels.append(f(x))
  # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=model_fn, model_dir="/tmp/converge_convnet_model")
  
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"value" : "prediction", "features": "features", "labels": "labels"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
  
    # Train the model
    mnist_classifier.fit(
        x=np.array(train_data, dtype=np.float32),
        y=np.array(train_labels, dtype=np.float32),
        batch_size=1,
        steps=20000,
        monitors=[logging_hook])
  
    # Configure the accuracy metric for evaluation
    metrics = {
        "mean_absolute_error":
            learn.MetricSpec(
                metric_fn=tf.metrics.mean_absolute_error, prediction_key="value"),
    }
  
    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=np.array(eval_data, dtype=np.float32), y=np.array(eval_labels, dtype=np.float32), metrics=metrics, batch_size=1)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
