# PAPRIKA: Private Online False Discovery Rate Control
A Python implementation of our paper PAPRIKA: Private Online False Discovery Rate Control

Documentation
===

Plots saved as .pdf files in the folder "plots", data saved as .dat files in the folder "dat".

Note that the plots may look different than the ones in the paper because the observations are randomly generated.

The main file is `run_and_plot.py`. The experiments vary depending on the following passed arguments:  
`--FDRrange` - integers encoding the choice of algorithms and parameters. 1 for SAFFRON, 2 for LORD, 3 for Alpha-investing, 4 for SAFFRON AI, 5 for PAPRIKA AI, 6 for PAPRIKA  
`--num-runs` - number of independent trials  
`--num-hyp` - number of hypotheses  
`--plot-style` - 0 for plots vs hyp, 1 for plots vs pi1, 2 for varying espilon, 3 for varying shift magnitude, 4 for varying signal in the truncated exponential distribution example  
`--alpha0` - test level  
`--mu-gap` - used for Gaussian tests as mu_c, where observations under the alternative are N(Z,1), Z~N(mu_c,1) (in SAFFRON paper)  
`--lbd-scale` - used for alternative signal in truncated exponential example  
`--mod-choice` - 1 for Gaussian tests, 2 for beta alternatives (in SAFFRON paper), 3 for truncated exponential example, 4 for Bernoulli example  
`--pirange` - range of pi1  
`--eps` - privacy parameter epsilon  
`--epsrange` - used for varying epsilon for private algorithms  
`--sensitivity` - eta in the multiplicative sensitivity, use np.sqrt(np.log(1000)/1000)  
`--shift`- shift magnitude c  
`--shiftrange`- used for varying shift magnitude for private algorithms  
`--lbdrange` - used for varying alternative signal in truncated exponential example  

This code borrowed substantial parts from SAFFRON code available at: https://github.com/tijana-zrnic/SAFFRONcode.  
Their code in turn relied heavily on code corresponding to the paper "Online control of the false discovery rate with decaying memory": https://github.com/fanny-yang/OnlineFDRCode.

Examples:

To make Figure 1 (a)(b) in the paper, run
```
python3  run_and_plot.py --FDRrange  1,2,3,4,5,6 
```
 
To make Figure  1 (c)(d) in the paper, run 
```
python3  run_and_plot.py --FDRrange  5 --plot-style 2 
```
 
To make Figure  1 (e)(f) in the paper, run 
```
python3  run_and_plot.py --FDRrange  6 --plot-style 2 
```

To make Figure 2 (a)(b) in the paper, run
```
python3  run_and_plot.py --FDRrange  1,2,3,4,5,6  --lbd-scale 1.95 --mod-choice 3 
```
 
To make Figure  2 (c)(d) in the paper, run 
```
python3  run_and_plot.py --FDRrange  5 --plot-style 2 --lbd-scale 1.95 --mod-choice 3 
```
 
To make Figure  2 (e)(f) in the paper, run 
```
python3  run_and_plot.py --FDRrange  6 --plot-style 2  --lbd-scale 1.95 --mod-choice 3 
```

To make Figure 3 (a)(b) in the paper, run
```
python3  run_and_plot.py --FDRrange  1,2,3,4,5,6  --plot-style 0 --lbd-scale 1.95 --mod-choice 3 
```

To make Figure 4 (a)(b) in the paper, run
```
python3  run_and_plot.py --FDRrange  5  --plot-style 4  --mod-choice 3 
```

To make Figure 4 (c)(d) in the paper, run
```
python3  run_and_plot.py --FDRrange  6  --plot-style 4  --mod-choice 3 
```

To make Figure 5 (a)(b) in the paper, run
```
python3  run_and_plot.py --FDRrange  5  --plot-style 3
```

To make Figure 5 (c)(d) in the paper, run
```
python3  run_and_plot.py --FDRrange  6  --plot-style 3
```

### Bibliographic Information

If you use our code or paper, we ask that you please cite:
```
@article{ZhangKC20,
  title         = {PAPRIKA: Private Online False Discovery Rate Control},
  author        = {Zhang, Wanrong and Kamath, Gautam and Cummings, Rachel},
  journal       = {arXiv preprint arXiv:2002.12321},
  year          = {2020}
}
```

