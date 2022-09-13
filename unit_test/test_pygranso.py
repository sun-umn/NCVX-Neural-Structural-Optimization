import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Experiments-PAMI')
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


from utils import models
from utils import problems
from utils import topo_api




# train CNN-LBFGS model
def train_cnn_model(problem, max_iterations, cnn_kwargs=None):
    args = topo_api.specified_task(problem)
    # model = models.CNNModel(args=args, **cnn_kwargs)
    model = models.CNNModel_torch(args=args, **cnn_kwargs)

    z = torch.randn(1,128)

    output = model(z)

    # [1,20,60] [batch_size,width,height]
    output = torch.squeeze(output,1)

    # ds_cnn = train.train_lbfgs(model, max_iterations)
    pass

problem = problems.mbb_beam(height=20, width=60)
ds = train_cnn_model(problem, max_iterations=10, cnn_kwargs=dict(resizes=(1, 1, 2, 2, 1)))


