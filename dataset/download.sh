#!/bin/bash

wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
kaggle datasets download -d praveengovi/credit-risk-classification-dataset
unzip credit-risk-classification-dataset.zip
rm credit-risk-classification-dataset.zip

kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip
rm gtsrb-german-traffic-sign.zip
rm -rf meta
rm -rf Meta
rm -rf test 
rm -rf Test
rm -rf train
rm -rf Train
rm Meta.csv
rm Test.csv
mv Train.csv gtsrb.csv

wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
mv agaricus-lepiota.data mushroom.data
wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data

kaggle datasets download -d kingburrito666/shakespeare-plays
unzip shakespeare-plays.zip
rm shakespeare-plays.zip
rm william-shakespeare-black-silhouette.jpg
rm alllines.txt 
