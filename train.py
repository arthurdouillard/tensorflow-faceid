import random
import os
import json

from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import trange

import faceid


n_iterations = 10000
step = 500
batch_size = 16
n_channels = 1#3
img_size = (100, 100)#(196, 196)
data = 'data/att_faces'#'data/faces96'
n_convs=[2, 3, 3]
n_filters=[16, 32, 64]


loader = faceid.loader.Loader1(
    data,
    batch_size=32,
    target_size=img_size,
    n_channels=n_channels
)


model = faceid.model.Model(
    img_size,
    n_convs=n_convs,
    n_filters=n_filters,
    n_channels=n_channels
)

if not os.path.exists('model'):
    os.mkdir('model')
with open('model/convnet.json', 'w+') as f:
    json.dump(
        {
            'n_convs': n_convs,
            'n_filters': n_filters,
            'n_channels': n_channels,
            'img_size': img_size
        },
        f
    )


total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print('nb params', total_parameters)

saver = tf.train.Saver(
    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'convnet*')
)

best_loss = float('inf')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(n_iterations // step):
        epoch_loss = 0.
        epoch_acc = 0.

        t = trange(step, desc='Metrics')
        for j in t:
            left, right, similarity = loader.get(batch_size)
            _, loss, acc = sess.run(
                [model.opt, model.loss, model.acc],
                feed_dict={
                    model.left: left,
                    model.right: right,
                    model.similarity: similarity
                }
            )

            epoch_loss += loss
            epoch_acc += acc

            t.set_description('Loss = {}; Acc = {}'.format(
                round(epoch_loss / (j+1), 4),
                round(epoch_acc / (j+1), 4)
            ))

            if np.isnan(loss):
                print('Nan loss!')
                exit(1)


        print('Iterations {}: Loss = {}; Acc = {}'.format(
                i * j,
                round(epoch_loss / (j+1), 4),
                round(epoch_acc / (j+1), 4)
        ))
        epoch_loss /= step
        epoch_acc /= step
        if epoch_loss < best_loss:
           print('Model improved, saving...', end=' ')
           saver.save(sess, './model/model')
           best_loss = epoch_loss
           print('Ok!')
