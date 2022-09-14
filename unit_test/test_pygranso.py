import time
import torch
import sys

# from neural_structural_optimization.topo_physics import get_stiffness_matrix
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Experiments-PAMI')
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


from utils import models
from utils import problems
from utils import topo_api
from utils import train




# train CNN-LBFGS model
def train_cnn_model(problem, max_iterations, cnn_kwargs=None):
    args = topo_api.specified_task(problem)
    # model = models.CNNModel(args=args, **cnn_kwargs)
    model = models.CNNModel_torch(args=args, **cnn_kwargs)

    z = torch.randn(1,128)

    output = model(z)

    # [1,20,60] [batch_size,width,height]
    output = torch.squeeze(output,1)

    # print(list(model.parameters())[0].shape)

    for name, param in model.named_parameters():
        print("{}: {}".format(name, param.data.shape))
    # ds_cnn = train.train_lbfgs(model, max_iterations)
    pass

def test(problem, max_iterations, cnn_kwargs=None):
    args = topo_api.specified_task(problem)
    model = models.CNNModel(args=args, **cnn_kwargs)


    ds_cnn = train.train_lbfgs(model, max_iterations)
    pass


def user_fn(model,z):
    x = torch.squeeze(model(z),1)
    U = list(model.parameters())[0]
    K = get_stiffness_matrix()

    # objective function
    f = U@K@U

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = V(x) - V0
    ce.c2 = nature_img_box_constr
    ce.c3 = K@U - F

    return [f,ci,ce]

problem = problems.mbb_beam(height=20, width=60)
ds = train_cnn_model(problem, max_iterations=10, cnn_kwargs=dict(resizes=(1, 1, 2, 2, 1)))
# ds = test(problem, max_iterations=10, cnn_kwargs=dict(resizes=(1, 1, 2, 2, 1)))


