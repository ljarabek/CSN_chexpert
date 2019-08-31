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

Feel free to report other known bugs.

This is a work in progress :)
