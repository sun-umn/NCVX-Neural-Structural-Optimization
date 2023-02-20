# stdlib
import gc
import random
import time
from os import path

# third party
import cvxopt
import cvxopt.cholmod
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# first party
import models
import topo_api
import topo_physics
import utils


def multi_material_volume_constrained_structural_optimization_function(
    model, initial_compliance, initial_mass, ke, args, device, dtype
):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.

    Notes:
    For the original MBB Beam the best alpha is 5e3
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x
    unscaled_compliance, full_x_phys = topo_physics.calculate_multi_material_compliance(
        model, ke, args, device, dtype
    )
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    mass_constraint_value = utils.compute_mass_constraint(full_x_phys, args)

    ce.c1 = mass_constraint_value / initial_mass  # noqa

    # # Let's try and clear as much stuff as we can to preserve memory
    del full_x_phys, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def train_mass_constrained_multi_material(
    problem,
    model_type,
    e_materials,
    num_materials,
    volfrac,
    combined_volfrac,
    aggregation,
    penal,
    seed=0,
):
    """
    Function to train the mass constrained multi-material problem with
    pygranso
    """
    # Set up seed for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For now setup the device to be CPU
    device = torch.device("cpu")

    # Get the args
    args = topo_api.specified_task(problem, device=device)

    # TODO: Once we get this example working we can configure the problem in the right
    # way but for now we will set it from the inputs to the function
    args = topo_api.specified_task(problem, device=device)
    args["e_materials"] = e_materials
    args["num_materials"] = num_materials
    args["volfrac"] = volfrac
    args["combined_volfrac"] = combined_volfrac
    args["aggregation"] = aggregation
    args["penal"] = penal

    # Get the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=device,
    ).to(dtype=torch.double)

    # Keyword arguments for the different types of models we will
    # allow. Currently, we have the DIP multi-material and the MLP
    # multi-material for comparison
    numLayers = 5  # noqa
    numNeuronsPerLyr = 20  # noqa
    cnn_kwargs = dict(resizes=(1, 2, 2, 2, 1))

    # Select the model
    if model_type == "cnn":
        model = models.MultiMaterialModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

    elif model_type == "mlp":
        model = models.TopNetPyGranso(
            numLayers,
            numNeuronsPerLyr,
            args["nelx"],
            args["nely"],
            args["num_materials"],
            symXAxis=False,
            symYAxis=False,
            seed=seed,
        ).to(device=device, dtype=torch.double)

    else:
        raise ValueError("There is no such model!")

    # Put the model in training mode
    model.train()

    # Calculate the inital compliance
    initial_compliance, x_phys = topo_physics.calculate_multi_material_compliance(
        model, ke, args, device, torch.double
    )
    initial_compliance = torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0

    # If we can get the initial compliance then we can also get the intial
    # mass constraint
    initial_mass_constraint = utils.compute_mass_constraint(x_phys, args)

    # Combined function
    comb_fn = lambda model: multi_material_volume_constrained_structural_optimization_function(  # noqa
        model,
        initial_compliance,
        initial_mass_constraint,
        ke,
        args,
        device=device,
        dtype=torch.double,
    )

    # Initalize the pygranso options
    opts = pygransoStruct()

    # Set the device
    opts.torch_device = device

    # Setup the intitial inputs for the solver
    nvar = getNvarTorch(model.parameters())
    opts.x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
    ).to(device=device)

    # Additional pygranso options
    opts.limited_mem_size = 20
    opts.torch_device = device
    opts.double_precision = True
    opts.mu0 = 1.0
    opts.maxit = 100
    opts.print_frequency = 1
    opts.stat_l2_model = False
    opts.viol_eq_tol = 1e-4
    opts.opt_tol = 1e-4
    opts.init_step_size = 1e-1

    # Main algorithm with logging enabled.
    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)

    return model, ke, args, soln


###### Code From MM Neural Network Paper #####
# Structural FE
class StructuralFE:
    def getDMatrix(self):
        E = 1
        nu = 0.3
        k = np.array(
            [
                1 / 2 - nu / 6,
                1 / 8 + nu / 8,
                -1 / 4 - nu / 12,
                -1 / 8 + 3 * nu / 8,
                -1 / 4 + nu / 12,
                -1 / 8 - nu / 8,
                nu / 6,
                1 / 8 - 3 * nu / 8,
            ]
        )
        KE = (
            E
            / (1 - nu**2)
            * np.array(
                [
                    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
                ]
            )
        )
        return KE

    # -----------------------#
    def initializeSolver(self, nelx, nely, forceBC, fixed, penal=3, elemArea=1):

        self.penal = penal
        self.elemArea = elemArea
        self.nelx = nelx
        self.nely = nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.KE = self.elemArea * self.getDMatrix()
        self.fixed = fixed
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)
        self.f = forceBC

        self.edofMat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array(
                    [
                        2 * n1 + 2,
                        2 * n1 + 3,
                        2 * n2 + 2,
                        2 * n2 + 3,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n1,
                        2 * n1 + 1,
                    ]
                )

        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()

    # -----------------------#
    def initializeMultiMaterial(self, EMaterials):
        self.numMaterials = EMaterials.shape[0]
        self.EMaterials = EMaterials
        self.Emax = max(EMaterials) * np.ones((self.nelx * self.nely))

    # -----------------------#
    def solve88(self, density):
        self.densityField = density
        self.u = np.zeros((self.ndof, 1))
        EnetOfElem = np.array(
            [
                np.dot(self.EMaterials, (density[i, :]) ** self.penal)
                for i in range(density.shape[0])
            ]
        )
        sK = ((self.KE.flatten()[np.newaxis]).T * (EnetOfElem)).flatten(order="F")
        self.sK = sK
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        K = self.deleterowcol(K, self.fixed, self.fixed).tocoo()
        K = cvxopt.spmatrix(K.data, K.row.astype(np.int), K.col.astype(np.int))
        B = cvxopt.matrix(self.f[self.free, 0])
        cvxopt.cholmod.linsolve(K, B)
        self.u[self.free, 0] = np.array(B)[:, 0]
        self.Jelem = (
            np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * self.u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)

        return self.u, self.Jelem

    # -----------------------#
    def deleterowcol(self, A, delrow, delcol):
        # Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete(np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete(np.arange(0, m), delcol)
        A = A[:, keep]
        return A

    # -----------------------#
    def computeElementStrains(self):
        delta = self.u[self.edofMat].reshape(self.nelx * self.nely, 8)
        s = np.sqrt(self.elemArea)
        strainU = 0.5 * (delta[:, 2] - delta[:, 0] + delta[:, 4] - delta[:, 6]) / s
        strainV = 0.5 * (-delta[:, 3] - delta[:, 1] + delta[:, 5] + delta[:, 7]) / s
        strainUV = 0.5 * (
            0.5 * (delta[:, 6] - delta[:, 0] + delta[:, 4] - delta[:, 2]) / s
            + 0.5 * (delta[:, 7] - delta[:, 1] + delta[:, 5] - delta[:, 3]) / s
        )
        return strainU, strainV, strainUV

    def plotFE(self):

        elemDisp = self.u[self.edofMat].reshape(self.nelx * self.nely, 8)
        elemU = (elemDisp[:, 0] + elemDisp[:, 2] + elemDisp[:, 4] + elemDisp[:, 6]) / 4
        elemV = (elemDisp[:, 1] + elemDisp[:, 3] + elemDisp[:, 5] + elemDisp[:, 7]) / 4
        rhoElem = np.array(
            [np.max(self.densityField[i, :]) for i in range(self.densityField.shape[0])]
        )
        strainU, strainV, strainUV = self.computeElementStrains()

        delta = np.sqrt(elemU**2 + elemV**2)

        scale = 0.1 * max(self.nelx, self.nely) / max(delta)

        x, y = np.mgrid[: self.nelx, : self.nely]
        x = x + scale * elemU.reshape(self.nelx, self.nely)
        y = y + scale * elemV.reshape(self.nelx, self.nely)
        z = delta.reshape(self.nelx, self.nely)

        # plot FE results
        fig = plt.figure()  # figsize=(10,10)
        plt.subplot(2, 2, 1)
        z = strainU.reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap="jet")
        plt.title("strain X")
        fig.colorbar(im)

        plt.subplot(2, 2, 2)
        z = strainV.reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap="jet")
        plt.title("strain Y")
        fig.colorbar(im)

        plt.subplot(2, 2, 3)
        z = delta.reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap="jet")
        plt.title("net deformation")
        fig.colorbar(im)

        plt.subplot(2, 2, 4)
        z = (rhoElem * self.Jelem).reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap="jet")
        plt.title("element compliance")
        fig.colorbar(im)

        fig, axes = plt.subplots(ncols=1)
        plt.pcolormesh(x, y, z, cmap="jet")


# Set devices
def setDevice(device):  # noqa
    devices = {"cpu": 0, "gpu": 1}
    if torch.cuda.is_available() and (devices[device] == 1):
        device = torch.device("cuda:0")
        print("GPU enabled")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device


# Set seed
def set_seed(manualSeed):  # noqa
    """
    Function to set the seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)


# Loss function
class TopOptLoss(nn.Module):
    def __init__(self):
        super(TopOptLoss, self).__init__()

    def forward(
        self,
        volFracNN,
        phiElem,
        desiredMass,
        densityOfMaterial,
        elemArea,
        penal,
        numMaterials,
    ):

        objective = torch.zeros(numMaterials)
        massMaterial = torch.zeros(numMaterials)

        for i in range(numMaterials):
            objective[i] = torch.sum(torch.div(phiElem[i], volFracNN[:, i] ** penal))
            # summed over all elems for the particular material
            massMaterial[i] = torch.sum(
                densityOfMaterial[i] * volFracNN[:, i] * elemArea
            )

        massConstraint = (torch.sum(massMaterial) / desiredMass) - 1.0
        return objective, massConstraint


class TopologyOptimizer:
    #%% topoOpt
    def initializeFE(
        self,
        problemName,
        nelx,
        nely,
        elemArea,
        forceBC,
        fixed,
        device="cpu",
        penal=3,
        nonDesignRegion=None,
        EMaterials=np.array([1.0]),
    ):
        self.problemName = problemName
        self.nelx = nelx
        self.nely = nely
        self.boundaryResolution = 1
        self.FE = StructuralFE()
        self.elemArea = elemArea
        self.FE.initializeSolver(nelx, nely, forceBC, fixed, penal, elemArea)
        self.FE.initializeMultiMaterial(EMaterials)
        self.device = setDevice(device)
        self.nonDesignRegion = nonDesignRegion
        self.xy, self.nonDesignIdx = self.generatePoints(nelx, nely, 1, nonDesignRegion)
        self.xyPlot, self.nonDesignPlotIdx = self.generatePoints(
            nelx, nely, self.boundaryResolution, nonDesignRegion
        )

    def generatePoints(
        self, nx, ny, resolution=1, nonDesignRegion=None
    ):  # generate points in elements
        ctr = 0
        xy = np.zeros((resolution * nx * resolution * ny, 2))
        nonDesignIdx = []
        for i in range(resolution * nx):
            for j in range(resolution * ny):
                xy[ctr, 0] = (i + 0.5) / resolution
                xy[ctr, 1] = (j + 0.5) / resolution
                if nonDesignRegion is not None:
                    if (
                        (xy[ctr, 0] < nonDesignRegion["x<"])
                        and (xy[ctr, 0] > nonDesignRegion["x>"])
                        and (xy[ctr, 1] < nonDesignRegion["y<"])
                        and (xy[ctr, 1] > nonDesignRegion["y>"])
                    ):
                        nonDesignIdx.append(ctr)
                ctr += 1
        xy = torch.tensor(xy, requires_grad=False).float().view(-1, 2).to(self.device)
        return xy, nonDesignIdx

    def initializeOptimizer(
        self,
        numLayers,
        numNeuronsPerLyr,
        model_type,
        desiredMassFraction,
        massDensityMaterials,
        symXAxis=False,
        symYAxis=False,
    ):
        self.desiredMassFraction = desiredMassFraction
        self.desiredMass = (
            desiredMassFraction
            * self.nelx
            * self.nely
            * np.max(massDensityMaterials)
            * self.elemArea
        )
        self.massDensityMaterials = massDensityMaterials
        self.density = np.zeros((self.FE.nelx * self.FE.nely, self.FE.numMaterials))
        self.density[:, np.argmax(self.massDensityMaterials)] = 1.0
        self.lossFunction = TopOptLoss()
        self.model_type = model_type

        if self.model_type == "mlp":
            self.topNet = models.TopNet(
                numLayers,
                numNeuronsPerLyr,
                self.FE.nelx,
                self.FE.nely,
                self.FE.numMaterials,
                symXAxis,
                symYAxis,
            ).to(self.device)

        elif self.model_type == "cnn":
            self.topNet = models.MultiMaterialCNN(
                self.FE.nelx,
                self.FE.nely,
                self.FE.numMaterials,
            ).to(self.device)

        self.objective = 0.0
        self.massConstraint = 0.0
        self.lossHistory = []
        self.topFig = plt.figure()
        self.topAx = self.topFig.gca()
        plt.ion()

    def setDesiredMass(
        self, desMass
    ):  # sometimes we wanna set desired mass instead of fraction
        self.desiredMass = desMass
        self.desiredMassFraction = desMass / (
            self.nelx * self.nely * np.max(self.massDensityMaterials) * self.elemArea
        )

    def train(self, maxEpochs, minEpochs, useSavedNet):
        self.convergenceHistory = []
        u, ce = self.FE.solve88(self.density)
        self.obj0 = np.array(
            [
                self.FE.EMaterials[i] * (self.density[:, i] ** self.FE.penal) * ce
                for i in range(self.FE.numMaterials)
            ]
        ).sum()

        self.convergenceHistory = self.optimizeDesign(maxEpochs, minEpochs, useSavedNet)
        # self.plotConvergence();
        return self.convergenceHistory

    #%% optimize design
    def optimizeDesign(self, maxEpochs, minEpochs, useSavedNet):
        lossHistory = []
        savedNetFileName = (
            "./results/"
            + self.problemName
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + ".nt"
        )
        if useSavedNet:
            if path.exists(savedNetFileName):
                self.topNet = torch.load(savedNetFileName)
            else:
                print("Network file not found")
        self.optimizer = optim.AdamW(
            self.topNet.parameters(), amsgrad=True, lr=0.001, weight_decay=1e-5
        )
        objHist = []
        alpha = 0.5
        alphaIncrement = 0.15
        nrmThreshold = 0.2
        self.numIter = 0
        batch_x = self.xy.to(self.device)

        for epoch in range(maxEpochs):
            self.optimizer.zero_grad()

            if self.model_type == "mlp":
                nnPred = self.topNet(batch_x, self.nonDesignIdx)[:, 1:].to(self.device)

            elif self.model_type == "cnn":
                nnPred = self.topNet(None)[:, 1:].to(self.device)

            #  return the pred dens for mat A/B ... (not void)
            nnPred_np = nnPred.cpu().detach().numpy()

            self.density = nnPred_np
            u, Jelem = self.FE.solve88(nnPred_np)
            self.phiElem = []
            for i in range(self.FE.numMaterials):
                self.phiElem.append(
                    torch.tensor(
                        self.FE.EMaterials[i]
                        * nnPred_np[:, i] ** (2 * self.FE.penal)
                        * Jelem
                        * self.elemArea
                    )
                    .view(-1)
                    .float()
                    .to(self.device)
                )
            objective, massConstraint = self.lossFunction(
                nnPred,
                self.phiElem,
                self.desiredMass,
                self.massDensityMaterials,
                self.elemArea,
                self.FE.penal,
                self.FE.numMaterials,
            )
            self.objective = torch.sum(objective) / self.obj0
            loss = self.objective + alpha * torch.pow(massConstraint, 2)

            loss.backward(retain_graph=True)
            #             torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            self.optimizer.step()
            self.massConstraint = massConstraint.detach().cpu().numpy()
            objHist.append(loss.item())

            # Get model weights
            # modelWeights, modelBiases = self.topNet.getWeights()
            relGreyElements = 1.0
            if (
                np.abs(self.massConstraint) < 0.2
            ):  # only comp relGray towards the end since its expensive operation!
                relGreyElements = (
                    sum(
                        sum(1 for rho in nnPred_np[:, i] if ((rho > 0.2) & (rho < 0.8)))
                        for i in range(self.FE.numMaterials)
                    )
                    / nnPred_np.shape[0]
                )
                relGreyElements /= nnPred_np.shape[1]
            self.FE.penal = min(4.0, self.FE.penal + 0.01)
            # continuation scheme
            lossHistory.append(
                [
                    self.massConstraint,
                    self.objective.item(),
                    loss.item(),
                    [np.mean(nnPred_np[:, i]) for i in range(self.FE.numMaterials)],
                ]
            )
            self.numIter = self.numIter + 1
            alpha = min(100, alpha + alphaIncrement)
            if epoch % 24 == 0:
                self.plotMaterialContour(epoch)
                print(
                    "{:d} p {:.3F} J0 {:.3F};  loss {:.3F}; massCons {:.3F}; relGray {:.3F}".format(
                        epoch,
                        self.FE.penal,
                        self.objective.item(),
                        loss.item(),
                        self.massConstraint,
                        relGreyElements,
                    )
                )
            if (
                (epoch > minEpochs)
                and (np.abs(self.massConstraint) < 0.05)
                and (relGreyElements < 0.035)
            ):
                print(
                    "{:d} p {:.3F} J0 {:.3F};  loss {:.3F}; massCons {:.3F}; relGray {:.3F}".format(
                        epoch,
                        self.FE.penal,
                        self.objective.item(),
                        loss.item(),
                        self.massConstraint,
                        relGreyElements,
                    )
                )
                break
        # self.FE.plotFE()
        torch.save(self.topNet, savedNetFileName)
        return lossHistory

    #%% plots

    def plotMaterialContour(
        self,
        iter,
        saveFig=False,
        fillColors=["0.90", "red", "cyan", "black", "pink", "blue"],
    ):
        plt.ion()
        plt.clf()

        x = self.xyPlot.cpu().detach().numpy()
        xx = np.reshape(
            x[:, 0],
            (
                self.boundaryResolution * self.FE.nelx,
                self.boundaryResolution * self.FE.nely,
            ),
        )
        yy = np.reshape(
            x[:, 1],
            (
                self.boundaryResolution * self.FE.nelx,
                self.boundaryResolution * self.FE.nely,
            ),
        )
        if self.model_type == "mlp":
            nnPred = self.topNet(self.xyPlot, self.nonDesignPlotIdx).to(self.device)
            nnPred_np = nnPred.detach().cpu().numpy()
            matIdx = np.array([np.argmax(rw) for rw in nnPred_np])

        elif self.model_type == "cnn":
            nnPred = self.topNet(None).to(self.device)
            nnPred_np = nnPred.detach().cpu().numpy()
            matIdx = np.array([np.argmax(rw) for rw in nnPred_np])

        a = plt.contourf(
            xx,
            yy,
            (0.01 + matIdx).reshape(
                (
                    self.boundaryResolution * self.FE.nelx,
                    self.boundaryResolution * self.FE.nely,
                )
            ),
            np.arange(0, self.FE.numMaterials + 2),
            colors=fillColors,
        )  # jet hatches= patterns,
        proxy = [
            plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
            for pc in a.collections
        ]
        plt.legend(
            proxy,
            np.arange(0, self.FE.numMaterials + 1),
            ncol=min(4, self.FE.numMaterials + 1),
        )

        self.topFig.canvas.draw()
        plt.title(
            "Iter = {:d}, J = {:.2F}, m/m*= {:.2F}".format(
                iter, self.objective * self.obj0, self.massConstraint + 1
            )
        )
        plt.axis("Equal")
        plt.pause(0.0001)
        if saveFig or iter % 24 == 0:
            plt.savefig("./frames/material_f_" + str(iter) + ".jpg")

    def plotMaterialImage(
        self,
        resolution=5,
        grids=False,
        fillColors=["0.90", "red", "cyan", "black", "pink", "blue"],
    ):
        xyPlot, nonDesignPlotIdx = self.generatePoints(
            self.FE.nelx, self.FE.nely, 1, self.nonDesignRegion
        )
        nnPred_np = self.topNet(xyPlot, nonDesignPlotIdx).detach().cpu().numpy()
        matIdx = np.array([np.argmax(rw) for rw in nnPred_np])
        fig, ax = plt.subplots()

        a = plt.imshow(
            np.flipud((matIdx).reshape((1 * self.FE.nelx, 1 * self.FE.nely)).T),
            cmap=colors.ListedColormap(fillColors),
            interpolation="none",
            vmin=0,
            vmax=self.FE.numMaterials + 1,
        )

        if grids:
            ax.xaxis.grid(True, zorder=0)
            ax.yaxis.grid(True, zorder=0)
            ax.set_xticks(
                np.arange(-0.50, resolution * (self.FE.nelx + 1) + 0.5, resolution)
            )
            ax.set_yticks(
                np.arange(-0.5, resolution * (self.FE.nely + 1) + 0.5, resolution)
            )
            ax.set_xticklabels(
                np.array(ax.get_xticks().tolist()) / resolution,
                fontsize=5,
                rotation="vertical",
            )
            ax.set_yticklabels(
                np.array(ax.get_yticks().tolist()) / resolution,
                fontsize=5,
                rotation="horizontal",
            )
            ax.axis("Equal")
            ax.grid(alpha=0.8)
            plt.grid(True)
        else:
            ax.axis("Equal")
            ax.axis("off")
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=clr) for clr in fillColors]
        plt.legend(
            proxy,
            np.arange(0, self.FE.numMaterials + 1),
            ncol=min(4, self.FE.numMaterials + 1),
        )

        netMass = (1 + self.massConstraint) * self.desiredMass
        plt.title(
            "J = {:.2F}; m = {:.2E}; iter = {:d} ".format(
                self.objective * self.obj0, netMass, self.numIter
            ),
            y=-0.15,
            fontsize="xx-large",
        )
        fName = (
            "./results/"
            + self.problemName
            + "_"
            + str(self.desiredMassFraction)
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + "_"
            + str(self.topNet.numLayers)
            + "_"
            + str(self.topNet.numNeuronsPerLyr)
            + "_topology.png"
        )
        fig.tight_layout()
        fig.savefig(fName, dpi=450)
        fig.show()

    def plotDensityGradient(self, saveFig=False):

        nnPred = self.topNet(self.xyPlot, self.nonDesignPlotIdx)
        x = self.xyPlot.cpu().detach().numpy()
        xx = np.reshape(
            x[:, 0],
            (
                self.boundaryResolution * self.FE.nelx,
                self.boundaryResolution * self.FE.nely,
            ),
        )
        yy = np.reshape(
            x[:, 1],
            (
                self.boundaryResolution * self.FE.nelx,
                self.boundaryResolution * self.FE.nely,
            ),
        )
        for i in range(self.FE.numMaterials + 1):
            fig = plt.figure()
            drho_dx = torch.autograd.grad(
                nnPred[:, i],
                self.xyPlot,
                grad_outputs=torch.ones(nnPred.shape[0]).to(self.device),
                create_graph=True,
            )[0]
            drho_dx = drho_dx.cpu().detach().numpy()
            normFactor = np.sqrt(drho_dx[:, 0] ** 2 + drho_dx[:, 1] ** 2)
            plt.contourf(
                xx,
                yy,
                -normFactor.reshape(
                    (
                        self.boundaryResolution * self.FE.nelx,
                        self.boundaryResolution * self.FE.nely,
                    )
                ),
                cmap=plt.cm.gray,
            )  # jet ,
            plt.title("Density gradient of material {:d}".format(i))
            fig.show()
            if saveFig or iter % 24 == 0:
                plt.savefig("./results/densityGrad_" + str(i) + ".jpg")

    def plotDensityContour(self, iter, saveFig=False):
        plt.ion()
        plt.clf()

        x = self.xyPlot.cpu().detach().numpy()
        xx = np.reshape(
            x[:, 0],
            (
                self.boundaryResolution * self.FE.nelx,
                self.boundaryResolution * self.FE.nely,
            ),
        )
        yy = np.reshape(
            x[:, 1],
            (
                self.boundaryResolution * self.FE.nelx,
                self.boundaryResolution * self.FE.nely,
            ),
        )
        nnPred = self.topNet(self.xyPlot, self.nonDesignPlotIdx).to(self.device)
        nnPred_np = nnPred.detach().cpu().numpy()
        E_rho_OfElem = np.array(
            [np.max(nnPred_np[i, 1:]) for i in range(nnPred_np.shape[0])]
        )

        a = plt.contourf(
            xx,
            yy,
            E_rho_OfElem.reshape(
                (
                    self.boundaryResolution * self.FE.nelx,
                    self.boundaryResolution * self.FE.nely,
                )
            ),
            cmap=plt.cm.rainbow,
        )  # jet
        plt.colorbar(a)
        self.topAx.axis("equal")
        self.topFig.canvas.draw()
        plt.title(
            "Iter = {:d}, J = {:.2F}, m/m*= {:.2F}".format(
                iter, self.objective * self.obj0, self.massConstraint + 1
            )
        )
        plt.pause(0.0001)
        if saveFig or iter % 24 == 0:
            fName = (
                "./results/"
                + self.problemName
                + "_contour_"
                + str(self.desiredMass)
                + "_"
                + str(self.nelx)
                + "_"
                + str(self.nely)
                + "_convergence.png"
            )
            plt.savefig(fName, dpi=450)

    def plotConvergence(self):
        self.convergenceHistory = np.array(self.convergenceHistory)
        plt.figure()
        plt.plot(self.convergenceHistory[:, 1], "b:", label="Rel.Compliance")
        plt.plot(self.convergenceHistory[:, 0], "r--", label="Mass constraint")
        plt.title(
            "Convergence Plots "
            + self.problemName
            + str(" m* = ")
            + str(self.desiredMassFraction)
        )
        plt.xlabel("Iterations")
        # plt.grid('on')
        plt.legend(loc="upper right", shadow=True, fontsize="large")
        fName = (
            "./results/"
            + self.problemName
            + "_"
            + str(self.desiredMass)
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + "_convergence.png"
        )
        plt.savefig(fName, dpi=450)

        print("Matidx \t E \t rho \t Vol(%) \t Mass \n")
        volFrac = np.zeros((self.FE.numMaterials + 1))
        netMass = 0.0
        for i in range(self.FE.numMaterials):
            vol = np.sum(self.density[:, i])
            volFrac[i] = vol / self.density.shape[0]
            mass = self.elemArea * vol * self.massDensityMaterials[i]
            netMass += mass
            print(
                "{:d} \t {:.2F} \t {:.2F} \t {:.1F} \t {:.1E}".format(
                    i + 1,
                    self.FE.EMaterials[i],
                    self.massDensityMaterials[i],
                    volFrac[i] * 100,
                    mass,
                )
            )
        print(
            " final J {:.2F} : , net mass {:.2F} ".format(
                self.objective * self.obj0, netMass
            )
        )


def runExample(model_type):

    nelx = 96
    nely = 32
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = 0.6
    #  perc of wt occupied by heaviest material
    #  ~~~~~~~~~~~~Material Parameters~~~~~~~~~~~~~#
    EMaterials = np.array([3.0, 2.0, 1.0])
    massDensityMaterials = np.array([1.0, 0.7, 0.4])
    #  ~~~~~~~~~~~~Simulation Parameters~~~~~~~~~~~~~#
    numLayers = 5
    # the depth of the NN
    numNeuronsPerLyr = 20
    # the height of the NN
    #  ~~~~~~~~~~~~Loading~~~~~~~~~~~~~#
    exampleName = "TipCantilever"
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = dofs[0 : 2 * (nely + 1) : 1]
    force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = False

    #  ~~~~~~~~~~~~Run code~~~~~~~~~~~~~#
    plt.close("all")
    minEpochs = 50
    maxEpochs = 2000
    penal = 1.0
    useSavedNet = False
    device = "cpu"
    # 'cpu' or 'gpu'

    start = time.perf_counter()
    topOpt = TopologyOptimizer()
    topOpt.initializeFE(
        exampleName,
        nelx,
        nely,
        elemArea,
        force,
        fixed,
        device,
        penal,
        nonDesignRegion,
        EMaterials,
    )
    topOpt.initializeOptimizer(
        numLayers,
        numNeuronsPerLyr,
        model_type,
        desiredMassFraction,
        massDensityMaterials,
        symXAxis,
        symYAxis,
    )
    # topOpt.setDesiredMass(450);
    lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)

    print("Time taken: {:.2F}".format(time.perf_counter() - start))
