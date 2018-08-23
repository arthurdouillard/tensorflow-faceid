#!/usr/bin/env python3
import argparse
import os
import glob
import json

import tensorflow as tf
import cv2
import numpy as np

import faceid


def load_model(args):
    params = json.load(open(os.path.join(args.model, 'convnet.json')))

    input_img = tf.placeholder(tf.float32, shape=[1, *params['img_size'], params['n_channels']])
    embed = faceid.model.convnet(input_img, params['n_convs'], params['n_filters'])

    sess = tf.Session()
    tf.train.Saver().restore(sess, os.path.join(args.model, 'model'))

    return input_img, embed, params['img_size'], sess


def cam_shooting(model, cache, threshold, haarcascade):
    face_cascade = cv2.CascadeClassifier(haarcascade)
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Face-id')

    while True:
        ret, frame = cam.read()
        if not ret: break

        frame_displayed = frame.copy()
        gray = cv2.cvtColor(frame_displayed, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        k = cv2.waitKey(1)

        if k % 256 == 27: # 'ESC'
            break

        for (x, y, w, h) in faces:
            name = get_name(model, cache, gray[y: y + h, x: x + w], threshold)

            if name == 'Unknown':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                frame_displayed,
                (x, y), (x + w, y + h),
                color, 2
            )
            cv2.putText(
                frame_displayed,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2
            )

        cv2.imshow('Face-id', frame_displayed)

    cam.release()
    cv2.destroyAllWindows()


def get_name(model, cache, img, threshold):
    input_img, embed, img_size, sess = model

    img = np.expand_dims(cv2.resize(img, tuple(img_size)), axis=-1)
    img = img / 127.5 - 1
    embeded_img = sess.run([embed], feed_dict={input_img: np.expand_dims(img, axis=0)})

    best_simi, best_name = float('inf'), 'Unknown'

    for name, npys in cache.items():
        for npy in npys:
            dist = np.linalg.norm(embeded_img - npy)

            if dist > threshold:
                continue
            elif dist < best_simi:
                best_simi = dist
                best_name = name

    return best_name


def load_cache():
    cache = {}

    for name in os.listdir('persons'):
        cache[name] = []

        for npy in glob.iglob(os.path.join('persons', name, '*.npy')):
            img = np.load(npy)
            cache[name].append(img)

    return cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faceid-like.')
    parser.add_argument('--model', type=str, help='Model folder.')
    parser.add_argument('--cam', type=str, help='Haarcascade.')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Threshold for face detection')
    args = parser.parse_args()

    model = load_model(args)
    cache = load_cache()

    cam_shooting(model, cache, args.threshold, args.cam)