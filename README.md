![](https://github.com/sun-umn/NCVX-Neural-Structural-Optimization/blob/master/topology-optimization.png)
# Topology Optimization
In this repository we explore the utilization of deep neural networks to power structure and topological optimization.

## Installation

conda create -n ncvx_exp_pami python=3.9

## Install pygranso

```bash
git clone https://github.com/sun-umn/PyGRANSO.git

cd PyGRANSO

pip install git+https://github.com/sun-umn/PyGRANSO@dimension_factor

pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## (Optional, not used in this src) Install neural-structral-optimization

```bash
pip install -q tf-nightly git+https://github.com/google-research/neural-structural-optimization.git
```

## Problem Description

![Structural_OPT](./structural_opt.png)

The PyGRANSO implementation is based on the MBB beam example of *neural-structral-optimization*. See section **MBB Beam (Figure 2 from paper)** of https://github.com/google-research/neural-structural-optimization/blob/master/notebooks/optimization-examples.ipynb for more details.

## Running the code
`tasks.py` is the main code module to run code. Our current results are the outputs of the function `run_multi_structure_pipeline`. Assuming, everything is setup correctly our results can be reproduced via the following code blocks:

```python
from tasks import run_multi_structure_pipeline

run_multi_structure_pipeline()
```

or with an MSI job:

```bash
sbatch jobs/multi_structure_outputs.slurm
```


