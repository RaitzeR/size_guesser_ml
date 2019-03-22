
import json

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.lib.io import file_io


# create the base pre-trained model
def build_model(nb_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    print "starting model compile"
    compile(model)
    print "model compile done"
    return model


def save(model, tags, prefix, job_dir):
    filename = prefix + ".h5"
    model.save_weights(prefix+".h5")
    model.save(filename)
    tags_json = json.dumps(tags)
    # serialize model to JSON
    model_json = model.to_json()
    with file_io.FileIO(filename, mode='r') as inputFile:
        with file_io.FileIO(job_dir + '/' + filename, mode='w+') as outFile:
            outFile.write(inputFile.read())
    with file_io.FileIO(prefix+".json", "r") as json_file:
        with file_io.FileIO(job_dir + '/' + prefix+".json", mode='w+') as outFile:
            outFile.write(model_json)
    with file_io.FileIO(prefix+"-labels.json", "r") as json_file:
        with file_io.FileIO(job_dir + '/' + prefix+"-labels.json", mode='w+') as outFile:
            outFile.write(tags_json)
    #with open(prefix+".json", "w") as json_file:
    #    json_file.write(model_json)
    #with open(prefix+"-labels.json", "w") as json_file:
    #    json.dump(tags, json_file)


def load(prefix):
    # load json and create model
    with open(prefix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(prefix+".h5")
    with open(prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
