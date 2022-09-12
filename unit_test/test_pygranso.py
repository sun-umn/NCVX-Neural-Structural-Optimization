import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Experiments-PAMI')
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train




# train CNN-LBFGS model
def train_cnn_model(problem, max_iterations, cnn_kwargs=None):
    args = topo_api.specified_task(problem)
    model = models.CNNModel(args=args, **cnn_kwargs)
    ds_cnn = train.train_lbfgs(model, max_iterations)
    pass

problem = problems.mbb_beam(height=20, width=60)
ds = train_cnn_model(problem, max_iterations=10, cnn_kwargs=dict(resizes=(1, 1, 2, 2, 1)))


