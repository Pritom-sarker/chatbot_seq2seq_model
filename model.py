import tensorflow as tf
import numpy as np
from sklearn.externals import joblib
import My_Dl_lib as mdl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time


x=joblib.load('data/final_input.pkl')
y=joblib.load('data/final_output.pkl')

x=np.array(x)
y=np.array(y)

char2num=joblib.load('data/char2num.pkl')


x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1

print(x_seq_length,y_seq_length)




#peramiter

batch_size =128
nodes = 150
embed_size = 100
bidirectional = True


# Model representation

tf.reset_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, y_seq_length), 'output') # must be start with <GO> char
targets = tf.placeholder(tf.int32, (None,y_seq_length), 'targets')# Real word convert in numbers with out <GO>

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2num), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2num), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)

date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    if not bidirectional:

        # Regular approach with LSTM units
        lstm_enc = tf.contrib.rnn.LSTMCell(nodes)
        _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

    else:

        # Using a bidirectional LSTM architecture instead
        enc_fw_cell = tf.contrib.rnn.LSTMCell(nodes)
        enc_bw_cell = tf.contrib.rnn.LSTMCell(nodes)

        ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_fw_cell,
                                                                                                   cell_bw=enc_bw_cell,
                                                                                                   inputs=date_input_embed,
                                                                                                   dtype=tf.float32)
        enc_fin_c = tf.concat((enc_fw_final.c, enc_bw_final.c), 1)
        enc_fin_h = tf.concat((enc_fw_final.h, enc_bw_final.h), 1)
        last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)

with tf.variable_scope("decoding") as decoding_scope:
    if not bidirectional:
        lstm_dec = tf.contrib.rnn.LSTMCell(nodes)
    else:
        lstm_dec = tf.contrib.rnn.LSTMCell(2 * nodes)

    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)

logits = tf.layers.dense(dec_outputs, units=len(char2num), use_bias=True)

# connect outputs to
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# # Accuracy
#
# accuracy = tf.metrics.accuracy(labels=tf.argmax(targets, 1),
#                                   predictions=tf.argmax(logits,1))


if __name__=="__main__":

    saver = tf.train.Saver(save_relative_paths=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    print("Training Element : {} ".format(len(X_train)))
    print("Test Element  : {} ".format(len(y_test)))

    sess.run(tf.global_variables_initializer())
    mdl._check_restore_parameters(sess, saver)

    a=[]
    epochs = 1000
    for epoch_i in range(epochs):
        start_time = time.time()
        for batch_i in range(0,int(len(X_train)/batch_size)):
            source_batch, target_batch=mdl.getBatch(batch_i, batch_size,X_train, y_train)
            _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                feed_dict = {inputs: source_batch,
                 outputs: target_batch[:, :-1], # without last element
                 targets: target_batch[:, 1:]}) # with out first element

            # accuracy = sess.run(accuracy,
            #                     feed_dict={inputs:X_test,targets:y_test[:,1:],outputs:y_test[:,:-1]})
            accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
            print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} % Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,

                                                               accuracy*100, time.time() - start_time))


           # print(batch_loss)
        print("----------------------->>Step",epoch_i,"\n\n")
        batch_logits=sess.run(logits,feed_dict={inputs:X_test,outputs:y_test[:,:-1]})
        target_batch=y_test
        accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
        a.append(accuracy)
        print("accuracy-> {} % for test set".format(accuracy*100))
        if (epoch_i % 10 == 0):

            saver.save(sess, 'final_model/my_test_model',global_step=epoch_i)
            print("------- >> Model saved")

        if (epoch_i%25==0):
            plt.clf()
            plt.plot(a)
            plt.title("Accuracy curve ")
            plt.xlabel("Accuracy")
            plt.xlabel("Iteration")

            plt.savefig("curve/accuracy for Train set-{}.jpg".format(len(X_train)))
