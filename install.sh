#!/bin/bash

set +e
set +x

# Keep current folder
rootfolder=${PWD}

if [ ! -d "venv" ]; then
    virtualenv venv
fi

echo "Activating virtualenv..."
source ./venv/bin/activate
echo "Setup virtualenv for project under ${PWD}/venv/bin/activate"

echo "Installing Shapeworld"
cd ..
if [ ! -d "Shapeworld" ]; then
    git clone --recursive https://github.com/AlexKuhnle/ShapeWorld.git
fi
pip install -e Shapeworld
echo "Shapeworld successfully installed..."

echo "Installing dependencies..."
pip install -e ${rootfolder}

cd ${rootfolder}

echo "Done with setup"