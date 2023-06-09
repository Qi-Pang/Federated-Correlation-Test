#!/bin/bash

python3.7 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

deactivate