#!/usr/bin/env bash

mkdir data
cd data

# Download license file
wget https://s3.eu-central-1.amazonaws.com/avg-projects/plant/LICENSE.txt

# Download checkpoints
wget https://s3.eu-central-1.amazonaws.com/avg-projects/plant/dataset.zip
echo "Unzip dataset.zip (this may take a while)... we will make this faster soon"
unzip -q dataset.zip
rm dataset.zip

cd ..
