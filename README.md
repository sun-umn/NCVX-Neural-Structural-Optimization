# Installation

conda create -n ncvx_exp_pami python=3.9

## install pygranso

    git clone https://github.com/sun-umn/PyGRANSO.git

    cd PyGRANSO

    pip install git+https://github.com/sun-umn/PyGRANSO@dimension_factor

    pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html

## (Optional) install structural optimization

pip install -q tf-nightly git+https://github.com/google-research/neural-structural-optimization.git

# Problem Description

![Structural_OPT](./structural_opt.png)

The PyGRANSO implementation is based on the MBB beam example of *neural-structral-optimization*. See section **MBB Beam (Figure 2 from paper)** of https://github.com/google-research/neural-structural-optimization/blob/master/notebooks/optimization-examples.ipynb for more details.



# TODO List

    1. Stiffness matrix *K* should be a function of grid density *x*. Current implementation uses a fixed constant matrix *K*.

    2. 
