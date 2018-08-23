#!/usr/bin/env python3
import argparse
import os
import json
import glob

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Face registration')
    parser.add_argument('name', type=str,
                        help='Name of the person to register.')
    parser.add_argument('--model', type=str,
                        help='Embed the images, need the model folder.')
    parser.add_argument('--cam', type=str,
                        help='Enable cam to register face, need haar cascade.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose mode.')

    return parser.parse_args()


def load_model(args):
    import tensorflow as tf
    import faceid

    params = json.load(open(os.path.join(args.model, 'convnet.json')))

    input_img = tf.placeholder(tf.float32, shape=[1, *params['img_size'], params['n_channels']])
    embed = faceid.model.convnet(input_img, params['n_convs'], params['n_filters'])

    sess = tf.Session()
    tf.train.Saver().restore(sess, os.path.join(args.model, 'model'))

    return input_img, embed, params['img_size'], sess


def cam_shooting(args):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Face-id')

    face_cascade = cv2.CascadeClassifier(args.cam)

    faces_to_save = []

    color = (255, 0, 0)
    while True:
        ret, frame = cam.read()
        if not ret: break

        frame_displayed = frame.copy()
        gray = cv2.cvtColor(frame_displayed, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame_displayed,
                (x, y), (x + w, y + h),
                color, 2
            )
        color = (255, 0, 0)

        cv2.imshow('Face-id', frame_displayed)


        k = cv2.waitKey(1)

        if k % 256 == 27: # 'ESC'
            break
        elif k % 256 == 32: # 'SPACE'
            for (x, y, w, h) in faces:
                img = gray[y:y + h, x:x + w]
                faces_to_save.append(img)
            color = (0, 0, 255)

    cam.release()
    cv2.destroyAllWindows()

    return faces_to_save


def save_faces(faces, name):
    if not os.path.exists(name) or not os.path.isdir(name):
        os.mkdir(name)

    existing_names = os.listdir(name)
    max_id = -1
    for x in existing_names:
        x = x.split('.')[0]
        if x.isdigit():
            max_id = max(max_id, int(x))

    for face in faces:
        max_id += 1

        cv2.imwrite(os.path.join(name, '{}.jpg'.format(max_id)), face)


def embed_faces(input_img, embed, img_size, sess, name,):
    for jpg in glob.iglob(os.path.join(name, '*.jpg')):
        if os.path.exists(jpg[:-3] + 'npy'):
            continue

        img = Image.open(jpg)
        img = img.resize(img_size)
        arr = np.asarray(img)

        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=-1)

        arr = arr / 127.5 - 1

        face_embed = sess.run(
            [embed],
            feed_dict={input_img: np.expand_dims(arr, axis=0)}
        )

        np.save(jpg[:-3] + 'npy', face_embed[0])


def main():
    args = parse_args()

    if not os.path.exists('persons') or not os.path.isdir('persons'):
        os.mkdir('persons')

    if args.cam:
        faces = cam_shooting(args)
        save_faces(faces, os.path.join('persons', args.name))

    if args.model:
        input_img, embed, img_size, sess = load_model(args)
        embed_faces(input_img, embed, img_size, sess, os.path.join('persons', args.name))


if __name__ == '__main__':
    main()