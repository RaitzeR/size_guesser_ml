import coremltools

output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# For the first argument, use the filename of the newest .h5 file in the notebook folder.
coreml_mnist = coremltools.converters.keras.convert(
    'body_twelvth_test.h5', input_names=['image'], output_names=['output'],
    class_labels=output_labels, image_input_names='image')
