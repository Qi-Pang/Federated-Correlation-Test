# Secure Federated Correlation Test and Entropy Estimation
This repository contains the evaluation code for the ICML'23 paper: *Secure Federated Correlation Test and Entropy Estimation*.

### To reproduce the evaluation results
First, set up the running environment with the setup script.
```bash
./setup.sh
```

Enter the python3.7 virtual environment.
```bash
source ./venv/bin/activate
```

Get the datasets ready.
```bash
cd ./dataset
./download.sh
cd ..
```

Evaluate multiplicative error.
```bash
cd ./src
python simulate.py --data='synthesize' --correlation='independent'
python simulate.py --data='synthesize' --correlation='linear'
python simulate.py --data='synthesize' --correlation='quadratic'
python simulate.py --data='synthesize' --correlation='logistic'
python simulate.py --data='customer'
python simulate.py --data='payment_1'
python simulate.py --data='payment_2'
python simulate.py --data='payment_3'
python simulate.py --data='adult'
python simulate.py --data='gtsrb_width'
python simulate.py --data='gtsrb_height'
python simulate.py --data='gtsrb_x1'
python simulate.py --data='gtsrb_y1'
python simulate.py --data='gtsrb_x2'
python simulate.py --data='gtsrb_y2'
python simulate.py --data='mushroom_2_4'
python simulate.py --data='mushroom_8_13'
python simulate.py --data='mushroom_14_18'
python simulate.py --data='mushroom_19_21'
python simulate.py --data='lymphography'
```

Client-side Computation Overhead. Figure 2 in the paper is generated by running the following command using Pydroid 3 on a OnePlus 8.
```bash
python mobile_benchmark.py
```

Feature Selection.
```bash
python Reuters.py --type='filter'
python Reuters.py --type='orig'
python Reuters.py --type='train'
```

Online False Discovery Rate Control.
Code used in this part is borrowed from https://github.com/wanrongz/PAPRIKA.
```bash
cd ../PAPRIKA/
python run_and_plot.py
```
