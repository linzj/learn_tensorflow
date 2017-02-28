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
batch_size = 1
num_steps = 25
forget_bias = 1.0
learning_rate = 1e-1
sample_step_size = 200

graph = tf.Graph()

def build_graph(is_training, forget_bias, batch_size, num_steps, keep_prob = 1.0):
   r_inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps))
   r_targets = _targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps))
   _targets = tf.one_hot(_targets, vocab_size, axis=2)
   with tf.device("/cpu:0"):
       embedding = tf.get_variable(
           "embedding", [vocab_size, hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev = 1 / math.sqrt(hidden_size)))
       inputs = tf.nn.embedding_lookup(embedding, r_inputs)
   def lstm_cell():
       return tf.contrib.rnn.BasicLSTMCell(
           hidden_size, forget_bias=forget_bias)
   attn_cell = lstm_cell
   # cell = tf.contrib.rnn.MultiRNNCell(
   #     [attn_cell() for _ in range(4)])
   cell = attn_cell()
   if is_training and keep_prob < 1:
       inputs = tf.nn.dropout(inputs, keep_prob)

   state = cell.zero_state(batch_size, tf.float32)

   #inputs = tf.unstack(inputs, num=num_steps, axis=1)
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
                train_op, _loss, train_inputs, _targets = build_graph(True, forget_bias=forget_bias, batch_size=batch_size, num_steps=num_steps, keep_prob=0.7)

        with tf.name_scope("Sample"):
            with tf.variable_scope("Model", reuse=True) as vscope:
                sample_inputs, sample_op = build_graph(False, forget_bias=1.0, num_steps=sample_step_size, batch_size=1)
    n = 0
    p = 0
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
        init = tf.global_variables_initializer()
        init.run()
        print("Initialized")
        while True:
            if p+batch_size+num_steps+1 >= len(data):
                p = 0 # go from start of data
            inputs = []
            targets = []
            for i in range(batch_size):
                inputs_e = [char_to_ix[ch] for ch in data[p+i:p+i+num_steps]]
                targets_e = [char_to_ix[ch] for ch in data[p+i+1:p+i+num_steps+1]]
                inputs.append(inputs_e)
                targets.append(targets_e)
            _, loss = session.run([train_op, _loss], feed_dict = {train_inputs: inputs,
                    _targets: targets})
            if n % 100 == 0:
                print 'iter %d, loss: %f' % (n, loss) # print progress
                start_index = random.randint(1, (data_size - sample_step_size) // num_steps) * num_steps
                inputs = []
                for i in range(batch_size):
                    inputs.append([char_to_ix[ch] for ch in data[start_index:start_index+sample_step_size]])
                sample_ix = session.run(sample_op, feed_dict = {sample_inputs: inputs})
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print '----\n %s \n----' % (txt, )
            n += 1
            p += num_steps

if __name__ == '__main__':
    main()
