from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

graph = tf.Graph()
def build_graph():
    with graph.as_default():
        init = tf.global_variables_initializer()
        inputs = tf.placeholder(tf.int32, shape=[16])
        ids = tf.placeholder(tf.int32, shape=[2, 2])
        embeddings = tf.reshape(inputs, [4, 4])
        return init, inputs, ids, tf.nn.embedding_lookup(embeddings, ids)

def main():
    init, inputs, ids, result = build_graph()
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
        init.run()
        print("Initialized")
        result_val = session.run([result], feed_dict = {inputs: [1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16],
                ids: [[1, 1], [2, 2]]})
        print("result: " + str(result_val))

if __name__ == '__main__':
    main()
