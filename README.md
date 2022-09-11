# Installation

conda create -n ncvx_exp_pami python=3.9

## install pygranso

git clone https://github.com/sun-umn/PyGRANSO.git
cd PyGRANSO
pip install git+https://github.com/sun-umn/PyGRANSO@dimension_factor

pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html

## install structural optimization

pip install -q tf-nightly git+https://github.com/google-research/neural-structural-optimization.git

