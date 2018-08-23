import tensorflow as tf

from . import loader


class Model:
    def __init__(self, img_size, n_convs=[2, 2, 3, 3], n_filters=[64, 128, 256, 512],
		lr=0.001, n_channels=1, margin=2.0):
        self.img_size = img_size
        self.lr = lr
        self.margin = margin

        self._build_model(n_convs, n_filters, n_channels)

    def _build_model(self, n_convs, n_filters, n_channels):
        self.left = tf.placeholder(tf.float32, shape=[None, *self.img_size, n_channels])
        self.right = tf.placeholder(tf.float32, shape=[None, *self.img_size, n_channels])
        self.similarity = tf.placeholder(tf.float32, shape=[None])

        left_embed = convnet(self.left, n_convs, n_filters)
        right_embed = convnet(self.right, n_convs, n_filters)

        self._build_contrastive_loss(left_embed, right_embed)
        self._build_acc()

    def _build_contrastive_loss(self, left, right):
        dist_sqr = tf.reduce_sum(tf.square(
            tf.subtract(left, right)
        ))
        dist = tf.sqrt(dist_sqr)

        # When images are dissimilar, push up to the margin:
        loss = self.similarity * tf.square(tf.maximum(0., self.margin - dist))
        # When images are similar, reduce distance between them:
        loss = loss + (1 - self.similarity) * dist_sqr

        self.loss = tf.reduce_mean(loss)
        self.dist = dist

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_acc(self):
        gt_than_margin = tf.cast(
            tf.maximum(tf.subtract(self.dist, self.margin), 0.0),
            dtype=tf.bool
        )

        acc = tf.cast(gt_than_margin, dtype=tf.int32)
        self.acc = tf.reduce_mean(
            tf.cast(tf.not_equal(acc, tf.cast(self.similarity, dtype=tf.int32)),
                    dtype=tf.float32)
        )


def convnet(x, n_convs, n_filters):
    with tf.variable_scope('convnet'):
        for block_id, (c, f) in enumerate(zip(n_convs, n_filters)):
            for conv_id in range(c):
                with tf.variable_scope('block{}/conv{}'.format(block_id, conv_id)):
                    x = tf.layers.conv2d(x, f, 3, reuse=tf.AUTO_REUSE)
                    x = tf.layers.batch_normalization(x, reuse=tf.AUTO_REUSE)
                    x = tf.nn.relu(x)

            with tf.variable_scope('block{}/maxpool'.format(block_id)):
                x = tf.layers.max_pooling2d(x, 2, 1)

        with tf.variable_scope('head'):
            x = tf.reduce_mean(x, [1, 2])
            x = tf.layers.dense(x, 1024, reuse=tf.AUTO_REUSE)
            x = tf.layers.batch_normalization(x, reuse=tf.AUTO_REUSE)
            x = tf.nn.relu(x)

        x = tf.nn.l2_normalize(x)

    return x
