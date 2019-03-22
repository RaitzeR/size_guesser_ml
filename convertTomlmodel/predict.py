# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on

from keras.models import load_model
from keras.preprocessing import image
from keras.models import Model, model_from_json
import numpy as np
import json
import os
import sys
import scipy

# dimensions of our images
img_width, img_height = 224, 224

n = 224

imagepath = sys.argv[1]

print imagepath








def load(prefix):
    # load json and create model
    with open(os.path.dirname(os.path.abspath(__file__))+"/"+prefix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(os.path.dirname(os.path.abspath(__file__))+"/"+prefix+".h5")
    with open(os.path.dirname(os.path.abspath(__file__))+"/"+prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

def evaluate(model, vis_filename=None):
    # predicting images
    img = image.load_img(vis_filename, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    Y_pred = model.predict(images, batch_size=128)
    print Y_pred
    # y_pred = np.argmax(Y_pred, axis=1)
    #
    # accuracy = float(np.sum(y_test==y_pred)) / len(y_test)
    # print "accuracy:", accuracy
    #
    # confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    # for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
    #     confusion[predicted_index, actual_index] += 1
    # print "rows are predicted classes, columns are actual classes"
    # for predicted_index, predicted_tag in enumerate(tags):
    #     print predicted_tag[:7],
    #     for actual_index, actual_tag in enumerate(tags):
    #         print "\t%d" % confusion[predicted_index, actual_index],
    #     print
    # if vis_filename is not None:
    #     bucket_size = 10
    #     image_size = n // 4 # right now that's 56
    #     vis_image_size = nb_classes * image_size * bucket_size
    #     vis_image = 255 * np.ones((vis_image_size, vis_image_size, 3), dtype='uint8')
    #     example_counts = defaultdict(int)
    #     for (predicted_tag, actual_tag, normalized_image) in zip(y_pred, y_test, X_test):
    #         example_count = example_counts[(predicted_tag, actual_tag)]
    #         if example_count >= bucket_size**2:
    #             continue
    #         image = dataset.reverse_preprocess_input(normalized_image)
    #         image = image.transpose((1, 2, 0))
    #         image = scipy.misc.imresize(image, (image_size, image_size)).astype(np.uint8)
    #         tilepos_x = bucket_size * predicted_tag
    #         tilepos_y = bucket_size * actual_tag
    #         tilepos_x += example_count % bucket_size
    #         tilepos_y += example_count // bucket_size
    #         pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
    #         vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size, :] = image
    #         example_counts[(predicted_tag, actual_tag)] += 1
    #     vis_image[::image_size * bucket_size, :] = 0
    #     vis_image[:, ::image_size * bucket_size] = 0
    #     scipy.misc.imsave(vis_filename, vis_image)

def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

print "loading neural network"
model, tags = load("body_thirteenth_test")
compile(model)
print "done"

print "compiling predictor function" # to avoid the delay during video capture.
_ = model.predict(np.zeros((1, n, n, 3), dtype=np.float32), batch_size=1)
print "done"

img = image.load_img(imagepath, target_size=(img_width, img_height))
square = np.expand_dims(img, axis=0)
#square = square.transpose((0, 3, 1, 2))
square = preprocess_input(square)

probabilities = model.predict(square, batch_size=1).flatten()
prediction = tags[np.argmax(probabilities)]
print prediction + "\t" + "\t".join(map(lambda x: "%.2f" % x, probabilities))
print prediction