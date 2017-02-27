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
def build_graph(keep_prob = 1.0):
    with graph.as_default():
        init = tf.global_variables_initializer()
        _inputs = tf.placeholder(tf.int32, shape=(seq_length))
        targets = tf.placeholder(tf.int32, shape=(seq_length))
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, _inputs)
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        if keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        state = cell.zero_state(seq_length, tf.float32)
        
        with tf.variable_scope("RNN"):
            for time_step in range(seq_length):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[time_step], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_sum(losses)
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        return init, train_op, loss, _inputs, targets
        

def main():
    init, train_op, loss, inputs, targets  = build_graph()
    n = 0
    p = 0
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
        init.run()
        print("Initialized")
        if p+seq_length+1 >= len(data) or n == 0: 
            p = 0 # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
        while True:
            _, loss = session.run([train_op, loss], feed_dict = {inputs: inputs,
                    targets: targets})
            if n % 100 == 0:
                print 'iter %d, loss: %f' % (n, loss) # print progress

if __name__ == '__main__':
    main()
