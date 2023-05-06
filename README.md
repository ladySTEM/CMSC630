# CMSC630

For Part 3, Part3_FeatureExtraction.ipynb should be run first. It utilizes the library ImageProcessing.py to perform image segmentation and feature extraction. This execution file will output the file 'dataset.csv'.

Part3_ProcessCSV.ipynb should be run second, and will take an input of a dataset file and a value for 'k' from the config file 'config_csv.json'. This execution file will perform 10-fold validation using KNN Clustering. The output is an average accuracy of the validation for a value of k. 


The file 'dataset.csv' is included and is the result from running Part3_FeatureExtraction.ipynb.
