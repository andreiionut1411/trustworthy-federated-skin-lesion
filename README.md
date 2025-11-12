# trustworthy-federated-skin-lesion

The first thing we need to do is ensure that all the images across all the datasets are similar and have the same processing done in order to try to reduce the domain shift between them. To apply the processing you need to run:

`python3 src/data_processing/preprocess.py --images /path/to/images --size 256`