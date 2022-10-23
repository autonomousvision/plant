#!/usr/bin/env bash

# Download checkpoints
wget https://s3.eu-central-1.amazonaws.com/avg-projects/plant/checkpoints.zip
unzip -q checkpoints.zip
rm checkpoints.zip
