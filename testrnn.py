import numpy as np
import math, random
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
batch_size = data_size / seq_length
learning_rate = 1e-1

graph = tf.Graph()
def build_graph(is_training, keep_prob = 1.0):
   r_inputs = _inputs = tf.placeholder(tf.int32, shape=seq_length * batch_size)
   _inputs = tf.reshape(_inputs, (batch_size, seq_length))
   r_targets = _targets = tf.placeholder(tf.int32, shape=seq_length * batch_size)
   _targets = tf.reshape(_targets, (batch_size, seq_length))
   _targets = tf.one_hot(_targets, vocab_size)
   with tf.device("/cpu:0"):
       embedding = tf.get_variable(
           "embedding", [vocab_size, hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev = 1 / math.sqrt(hidden_size)))
       inputs = tf.nn.embedding_lookup(embedding, _inputs)
   cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
   if is_training and keep_prob < 1:
       inputs = tf.nn.dropout(inputs, keep_prob)

   state = cell.zero_state(batch_size, tf.float32)

   #inputs = tf.unstack(inputs, num=seq_length, axis=1)
   outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                              initial_state=state)

   output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
   softmax_w = tf.get_variable(
       "softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
   softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
   logits = tf.matmul(output, softmax_w) + softmax_b
   losses = tf.nn.softmax_cross_entropy_with_logits(labels=_targets, logits=logits)
   loss = tf.reduce_sum(losses)
   sample_op = tf.argmax(logits, 1, name="sample_argmax")
   if not is_training:
       return r_inputs, sample_op
   train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
   return train_op, loss, r_inputs, r_targets

def main():
    with graph.as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                train_op, _loss, train_inputs, _targets = build_graph(True, 0.7)

        with tf.name_scope("Sample"):
            with tf.variable_scope("Model", reuse=True) as vscope:
                sample_inputs, sample_op = build_graph(False)
    n = 0
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
        init = tf.global_variables_initializer()
        init.run()
        print("Initialized")
        total_size = batch_size * seq_length
        inputs = [char_to_ix[ch] for ch in data[:total_size]]
        targets = [char_to_ix[ch] for ch in data[1:total_size + 1]]
        print 'batch_size: %d' % batch_size
        while True:
            _, loss = session.run([train_op, _loss], feed_dict = {train_inputs: inputs,
                    _targets: targets})
            if n % 100 == 0:
                print 'iter %d, loss: %f' % (n, loss) # print progress
                start_index = random.randint(1, batch_size) * seq_length
                sample_inputs_data = data[start_index:total_size] + data[0:start_index]
                inputs = [char_to_ix[ch] for ch in sample_inputs_data]
                sample_ix = session.run(sample_op, feed_dict = {sample_inputs: inputs})
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print '----\n %s \n----' % (txt, )
            n += 1

if __name__ == '__main__':
    main()
