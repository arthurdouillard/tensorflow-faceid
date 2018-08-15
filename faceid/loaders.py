import os
import random

import numpy as np
import tensorflow as tf

class Loader:
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

        for i, person in enumerate(os.listdir(path)):
            person_path = os.path.join(path, person)
            mapping[person] = i

            for img in os.listdir(person_path):
                x.append(os.path.join(person_path, img))
                y.append(i)

        return np.array(x), np.array(y), mapping

    def _build_dataset(self):
        def sampler():

            cls1 = random.choice(range(0, self.n_classes))
            if random.random() < self.same_ratio: # Picking same class.
                paths = np.random.choice(
                    self.x[np.where(self.y == cls1)[0]],
                    size=2,
                    replace=False
                )
                similarity = 1.0
            else:
                cls2 = random.choice(range(0, cls1) + range(cls1 + 1, self.n_classes))
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

            return {'paths': paths, 'similarity': np.array(similarity)}

        def transform(data):
            paths = data['paths']

            img_left = tf.image.decode_image(paths[0])
            img_right = tf.image.decode_image(paths[0])

            img_left = tf.image.resize_images(img_left, self.target_size)
            img_right = tf.image.resize_images(img_right, self.target_size)

            return {
                'im_left': img_left,
                'img_right': img_right,
                'similarity': data['similarity']
            }

        dataset = tf.data.Dataset.from_generator(
            sampler,
            output_types={'paths': tf.string, 'similarity': tf.float16},
            output_shapes={'paths': tf.TensorShape([*self.target_size, 3]),
                           'similarity': tf.TensorShape([1])}
        )
        dataset = dataset.map(transform, num_parallel_calls=self.n_threads)
        dataset = dataset.prefetch(self.queue_size)
        dataset = dataset.repeat()
        self.dataset = dataset.batch(self.batch_size)

    def __next__(self):
        return self.iterator.get_next()
