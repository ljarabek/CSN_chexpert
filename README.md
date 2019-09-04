# CSN_chexpert
Experiments with adaptive instance normalization and tanh on chexpert and mimic

Uncertainties are labeled as negative.

Python 3.7
requirements.txt

## HOW TO USE:

Main file is main.py;

1. prepare CheXpert and MIMIC datasets in a directory - look at csv's in /data/2Label
	--> set arguement data_root to the path

2. set the lines 3,4 of the main.py accordingly if you use multiple GPUs. Default is for single-gpu. 
	!!! Saving plots won't work in multi-gpu mode

3. for information on other arguments, look at main.py or use the help command :)

4. run main.py

## DEBUGGING:
If you get out of range in pandas iloc method, it is because the labels.npy is not suitable for the dataset. Just delete labels.npy and wait for it to generate a new one.

One more bug is in the .png generation that produces chromatic aberration-like artifacts (blue).

Feel free to report other known bugs.

This is a work in progress.


# Results

CSN converges such that the input to densenet is actually binarized image with per-image learned treshold.
Currently the training is very unstable, CSN-Densenet best result after 30 epochs:

ROCAUC for Atelectasis is 0.7397375328083989

ROCAUC for Cardiomegaly is 0.8258689839572192

ROCAUC for Consolidation is 0.8818014705882353

ROCAUC for Edema is 0.8641369047619047

ROCAUC for Pleural Effusion is 0.8516757246376812


Multichannel version is more stable - the results below are only after the 2nd epoch. Models are from file models_multich.py:

ROCAUC for Atelectasis is 0.74498687664042

ROCAUC for Cardiomegaly is 0.8731060606060607

ROCAUC for Consolidation is 0.9137867647058824

ROCAUC for Edema is 0.9276785714285715

ROCAUC for Pleural Effusion is 0.9199501811594203


Compared to baseline - just Densenet. All same settings, except args.CSN = False. Converges and starts overfitting after 4 epochs.:

ROCAUC for Atelectasis is 0.6762204724409449

ROCAUC for Cardiomegaly is 0.7776292335115865

ROCAUC for Consolidation is 0.8880514705882354

ROCAUC for Edema is 0.8622023809523809

ROCAUC for Pleural Effusion is 0.8609601449275363

# Images

The single channel version examples:
Densenet achieves above results with the bottom left image as input.
The 2 numbers above the bottom right histogram are beta and gamma (shift and scale) from AdaIN repspectively. Images are binarized, since the AdaIN output is passed through a tanh before being fed into the model.

![alt text](https://raw.githubusercontent.com/ljarabek/CSN_chexpert/master/images/batch6_epoch_0_val.png)

![alt text](https://raw.githubusercontent.com/ljarabek/CSN_chexpert/master/images/batch7_epoch_0_val.png)

![alt text](https://raw.githubusercontent.com/ljarabek/CSN_chexpert/master/images/batch8_epoch_0_val.png)


# Other datasets

Experiments were performed on CIFAR100 dataset. CSN was adapted to have per channel outputs (RGB). There is no improvement on the densenet baseline, even when training CSN with smaller learning rate than densenet for improved stability. Furhter experiments are being made to improve CIFAR100 performance.
