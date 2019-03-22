# size_guesser_ml
Creates a model for a size categorisator. Essentially it just guesses the clothing size of a person. This is just for generating the model! Also you need your own dataset for it

## convertTomlmodel Folder
This folder contains scripts to convert the generated model to a .mlmodel - Note: Because of huge bugs in coremltools at the time of writing those scripts, I had to heavily modify the coremltools conversion scripts. So this might or might not work.

## trainer Folder
Google ML Engine configured ml model trainer. Dataset needs to be in a bucket called "body-size-ml-data" with folder structure like "XL" containing XL size training data, "M" containing M size training data etc.

Modify cloudml-gpu.yaml to change the hardware needed during training.
