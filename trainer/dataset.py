import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc
from google.cloud import storage
from tensorflow.python.lib.io import file_io


def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x


def reverse_preprocess_input(x0):
    x = x0 / 2.0
    x += 0.5
    x *= 255.
    return x


def dataset(base_dir, n):
    d = defaultdict(list)
    client = storage.Client()
    bucket = client.bucket('body-size-ml-data')
    subdirs = [
        "1X",
        "2X",
        "L",
        "M",
        "S",
        "XL",
        "XS"
    ]
    print "Going through sub directories"
    i = 0
    for subdir in subdirs:
        for filename in bucket.list_blobs(prefix=base_dir + "/" + subdir):
            if i < 10:
                print "gs://body-size-ml-data" + "/" + filename.name
            d[subdir].append("gs://body-size-ml-data" + "/" + filename.name)
            i = i + 1

    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
            file = file_io.FileIO(filename, mode='r')
            img = scipy.misc.imread(file)
            height, width, chan = img.shape
            assert chan == 3
            aspect_ratio = float(max((height, width))) / min((height, width))
            if aspect_ratio > 2:
                continue
            # We pick the largest center square.
            centery = height // 2
            centerx = width // 2
            radius = min((centerx, centery))
            img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
            img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
            X.append(img)
            y.append(class_index)
            useful_image_count += 1
            
    print "processed %d, used %d" % (processed_image_count, useful_image_count)

    X = np.array(X).astype(np.float32)
    #X = X.transpose((0, 3, 1, 2))
    X = preprocess_input(X)
    y = np.array(y)

    #print(X.shape)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print "classes:"
    for class_index, class_name in enumerate(tags):
        print class_name, sum(y==class_index)
    print
    return X, y, tags


def main():
    in_prefix, n = sys.argv[1:]
    X, y, tags = dataset(sys.stdin, in_prefix, n)
    print X.shape


if __name__ == "__main__":
    main()
