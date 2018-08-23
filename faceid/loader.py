import os
import random

import numpy as np
import numpy.random as rng
import tensorflow as tf
from PIL import Image



class Loader1:
    def __init__(self, path, batch_size=16, target_size=(224, 224), n_channels=1):
        self.x, self.y, self.mapping = self._parse_dataset(path)
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_channels = n_channels

        self.inv_mapping = {v:k for k,v in self.mapping.items()}
        self.n_classes = len(self.mapping)

    @staticmethod
    def open_image(path, target_size):
        img = Image.open(path)
        img = img.resize(target_size)
        arr = np.asarray(img)

        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=-1)

        return arr / 127.5 - 1

    def get(self, batch_size=None):
        batch_size = batch_size or self.batch_size

        x = np.empty((2, batch_size, *self.target_size, 1))
        y = np.zeros((batch_size))

        idx = 0
        classes = rng.choice(
            self.n_classes,
            batch_size,
            replace=False
        )
        # Same classes
        for cls in classes[:batch_size // 2]:
            try:
                paths = rng.choice(self.x[np.where(self.y == cls)[0]], 2)
            except:
                print(cls)
                print(np.where(self.y == cls))
            x[0, idx] = self.open_image(paths[0], self.target_size)
            x[1, idx] = self.open_image(paths[1], self.target_size)
            idx += 1

        # Different classes
        for cls in classes[batch_size // 2: batch_size]:
            left = rng.choice(self.x[np.where(self.y == cls)[0]], 1)[0]
            right = rng.choice(self.x[np.where(self.y != cls)[0]], 1)[0]

            x[0, idx] = self.open_image(left, self.target_size)
            x[1, idx] = self.open_image(right, self.target_size)
            y[idx] = 1
            idx += 1

        idxes = rng.permutation(batch_size)
        x = x[:, idxes, :, :, :]
        y = y[idxes]

        return x[0], x[1], y


    @staticmethod
    def _parse_dataset(path):
        x, y = [], []
        mapping = {}

        i = 0
        for person in os.listdir(path):
            if person.startswith('.'):
                continue

            person_path = os.path.join(path, person)
            imgs = os.listdir(person_path)

            if len(list(filter(lambda x: not x.startswith('.'), imgs))) == 0:
                continue

            mapping[person] = i

            for img in imgs:
                if img.startswith('.'):
                    continue
                x.append(os.path.join(person_path, img))
                y.append(i)

            i += 1

        return np.array(x), np.array(y), mapping

class Loader2:
    def __init__(self, path, queue_size=5, n_threads=4, batch_size=16,
                 same_ratio=0.5, target_size=(224, 224)):
        self.queue_size = queue_size
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.same_ratio = same_ratio
        self.target_size = target_size

        self.x, self.y, self.mapping = self._parse_dataset(path)
        self.inv_mapping = {v:k for k,v in self.mapping.items()}
        self.n_classes = len(self.mapping)

        self._build_dataset()
        self.iterator = self.dataset.make_one_shot_iterator()

    @staticmethod
    def _parse_dataset(path):
        x, y = [], []
        mapping = {}

        i = 0
        for person in os.listdir(path):
            if person.startswith('.'):
                continue

            person_path = os.path.join(path, person)
            mapping[person] = i

            for img in os.listdir(person_path):
                x.append(os.path.join(person_path, img))
                y.append(i)

            i += 1

        return np.array(x), np.array(y), mapping

    def _build_dataset(self):
        def sampler():
            cls1 = random.choice(list(range(0, self.n_classes)))
            if random.random() < self.same_ratio: # Picking same class.
                paths = np.random.choice(
                    self.x[np.where(self.y == cls1)[0]],
                    size=2,
                    replace=False
                )
                similarity = 1.0
            else:
                cls2 = random.choice(list(range(0, cls1)) + list(range(cls1 + 1, self.n_classes)))
                path1 = np.random.choice(
                    self.x[np.where(self.y == cls1)[0]],
                    size=1
                )

                path2 = np.random.choice(
                    self.x[np.where(self.y == cls2)[0]],
                    size=1
                )
                similarity = 0.0

                paths = np.array([path1, path2])

            return {
                'img_left': paths[0],
                'img_right': paths[1],
                'similarity': similarity
            }

        def transform(data):
            img_left = tf.image.decode_jpeg(data['img_left'], channels=3)
            img_right = tf.image.decode_jpeg(data['img_right'], channels=3)

            img_left = tf.image.resize_images(img_left, self.target_size)
            img_right = tf.image.resize_images(img_right, self.target_size)

            return {
                'img_left': img_left,
                'img_right': img_right,
                'similarity': data['similarity']
            }

        dataset = tf.data.Dataset.from_generator(
            sampler,
            output_types={
                'img_left': tf.string,
                'img_right': tf.string,
                'similarity': tf.float16
            },
            output_shapes={
                'img_left': tf.TensorShape([]),
                'img_right': tf.TensorShape([]),
                'similarity': tf.TensorShape([])
            }
        )
        dataset = dataset.map(transform, num_parallel_calls=self.n_threads)
        dataset = dataset.prefetch(self.queue_size)
        dataset = dataset.repeat()
        self.dataset = dataset.batch(self.batch_size)

    def get(self):
        return self.iterator.get_next()
