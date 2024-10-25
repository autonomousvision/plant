#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://tiny.carla.org/carla-0-9-10-linux
wget https://tiny.carla.org/additional-maps-0-9-10-linux
tar -xf carla-0-9-10-linux
tar -xf additional-maps-0-9-10-linux
rm carla-0-9-10-linux
rm additional-maps-0-9-10-linux
cd ..