import tensorflow as tf 
from  tensorflow.examples.tutorials.mnist  import  input_data
import numpy as np 

mnist = input_data.read_data_sets('data', one_hot = True)

# Here are the relevant parameters
num_classes = 10
input_size = 784
hidden_units_size = 30
batch_size = 100
training_iterations = 10000
parameter_path = "checkpoint/variable.ckpt"

# Here you set the structure of the input and output
X = tf.placeholder (tf.float32, shape = [None, input_size])
Y = tf.placeholder (tf.float32, shape = [None, num_classes])

# So here we have the network structure, and now we have two layers
W1 = tf.Variable (tf.random_normal ([input_size, hidden_units_size], stddev = 0.1))
B1 = tf.Variable (tf.constant (0.1), [hidden_units_size])
W2 = tf.Variable (tf.random_normal ([hidden_units_size, num_classes], stddev = 0.1))
B2 = tf.Variable (tf.constant (0.1), [num_classes])

hidden_opt = tf.matmul (X, W1) + B1
hidden_opt = tf.nn.relu (hidden_opt)
final_opt = tf.matmul (hidden_opt, W2) + B2
final_opt = tf.nn.relu (final_opt)

# Here you can set the type of the loss function.
loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels = Y, logits = final_opt))
opt = tf.train.GradientDescentOptimizer (0.05).minimize (loss)
init = tf.global_variables_initializer ()
correct_prediction = tf.equal (tf.argmax (Y, 1), tf.argmax (final_opt, 1))
accuracy = tf.reduce_mean (tf.cast (correct_prediction, 'float'))

sess = tf.Session ()
sess.run (init)
for i in range (training_iterations) :
    batch = mnist.train.next_batch (batch_size)
    batch_input = batch[0]
    batch_labels = batch[1]
    training_loss = sess.run ([opt, loss], feed_dict = {X: batch_input, Y: batch_labels})
    # prompt every 1000 steps
    if i % 1000 == 0 :
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: batch_input,Y: batch_labels})
        print ("step : %d, training accuracy = %g " % (i, train_accuracy))
    # To save weight
    saver.save(sess, parameter_path)
