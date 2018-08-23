# Face-ID

Goal: detect unique face identity.

How: Simple CNN made of `[[Conv - BN - ReLU]+ MaxPool]+`, followed by a
l2-normalization and a contrastive loss [(hadsell-chopra-lecun-06)](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf).

# Register your face:

1. Take pictures of your face:

```sh
./register my_name --cam data/haarcascade_frontalface_default.xml
```

A screen will appear, streaming your webcam feed. Make sure your head is detected
by OpenCv (in a blue box), and press `SPACE` to take a picture. Take several, from
different point of view.

Press `ESC` when you're done.


2. Compute the cached embedding:

```sh
./register my_name --model model/
```

Note that you can do step 1. and step 2. in a single command:

```sh
./register my_name --cam data/haarcascade_frontalface_default.xml --model model/
```

# Test it

```sh
./test.py --model model --cam data/haarcascade_frontalface_default.xml
```