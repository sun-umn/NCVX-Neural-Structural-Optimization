# stdlib
import random
from os import path

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import colors

from TOuNN.FE import StructuralFE

# from matplotlib import cm
plt.rcParams['figure.dpi'] = 150


# set device CPU/GPU
def setDevice(overrideGPU=True):  # noqa
    """
    Function to set the device for training
    """
    if torch.cuda.is_available() and (overrideGPU == False):  # noqa
        device = torch.device("cuda:0")
        print("GPU enabled")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device


overrideGPU = False
device = setDevice(overrideGPU)
torch.autograd.set_detect_anomaly(True)


# Seeding
def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)


def calculate_binary_constraint(design, epsilon):
    """
    Function to compute the binary constraint
    """
    return np.round(np.mean(design * (1 - design)) - epsilon, 4)


def calculate_volume_constraint(design, volume):
    """
    Function that computes the volume constraint
    """
    return np.round(np.mean(design) / volume - 1.0, 4)


# Neural network
class TopNet(nn.Module):
    inputDim = 2
    # x and y coordn of the point
    outputDim = 2
    # if material/void at the point

    def __init__(self, numLayers, numNeuronsPerLyr, nelx, nely, symXAxis, symYAxis):
        self.nelx = nelx
        # to impose symm, get size of domain
        self.nely = nely
        self.symXAxis = symXAxis
        # set T/F to impose symm
        self.symYAxis = symYAxis
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = self.inputDim
        manualSeed = 1234
        # NN are seeded manually
        set_seed(manualSeed)
        for lyr in range(numLayers):  # define the layers
            l = nn.Linear(current_dim, numNeuronsPerLyr)  # noqa
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = numNeuronsPerLyr
        self.layers.append(nn.Linear(current_dim, self.outputDim))
        self.bnLayer = nn.ModuleList()
        for lyr in range(numLayers):  # batch norm
            self.bnLayer.append(nn.BatchNorm1d(numNeuronsPerLyr))

    def forward(self, x, fixedIdx=None):
        # LeakyReLU ReLU6 ReLU
        m = nn.ReLU6()
        # LeakyReLU
        ctr = 0
        if self.symYAxis:
            xv = 0.5 * self.nelx + torch.abs(x[:, 0] - 0.5 * self.nelx)
        else:
            xv = x[:, 0]
        if self.symXAxis:
            yv = 0.5 * self.nely + torch.abs(x[:, 1] - 0.5 * self.nely)
        else:
            yv = x[:, 1]

        x = torch.transpose(torch.stack((xv, yv)), 0, 1)
        for layer in self.layers[:-1]:  # forward prop
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        out = 0.01 + torch.softmax(self.layers[-1](x), dim=1)
        # output layer
        rho = out[:, 0].view(-1)
        # grab only the first output

        rho = (1 - fixedIdx) * rho + fixedIdx * (rho + torch.abs(1 - rho))

        return rho

    def getWeights(self):  # stats about the NN
        modelWeights = []
        modelBiases = []
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.view(-1).cpu().numpy())
            modelBiases.extend(lyr.bias.data.view(-1).cpu().numpy())
        return modelWeights, modelBiases


# %%  compute loss
class TopOptLoss(nn.Module):
    def __init__(self):
        super(TopOptLoss, self).__init__()

    def forward(self, nn_rho, Jelem, desiredVolumeFraction, penal, obj0):
        # Scaled compliance
        objective = torch.sum(torch.div(Jelem, nn_rho**penal)) / obj0

        # Unscaled compliance
        volConstraint = (torch.mean(nn_rho) / desiredVolumeFraction) - 1.0
        return objective, volConstraint


# %% main TO functionalities
class TopologyOptimizer:
    def initializeFE(
        self,
        exampleName,
        nelx,
        nely,
        forceBC,
        fixed,
        penal=3,
        nonDesignRegion=None,
        Emin=1e-6,
        Emax=1.0,
        args=None,
    ):
        self.exampleName = exampleName
        self.nelx = nelx
        self.nely = nely
        self.boundaryResolution = 5
        # default value for plotting and interpreting
        self.FE = StructuralFE()
        self.FE.initializeSolver(nelx, nely, forceBC, fixed, penal, Emin, Emax)
        self.nonDesignRegion = nonDesignRegion
        self.xy, self.nonDesignIdx = self.generatePoints(nelx, nely, 1, nonDesignRegion)
        self.xyPlot, self.nonDesignPlotIdx = self.generatePoints(
            nelx, nely, self.boundaryResolution, nonDesignRegion
        )
        self.args = args

    def generatePoints(
        self, nx, ny, resolution=1, nonDesignRegion=None
    ):  # generate points in elements
        ctr = 0

        # NOTE: Changed some of the varables here becuase they were
        # calling global variables which is bad practice
        xy = np.zeros((resolution * nx * resolution * ny, 2))
        nonDesignIdx = torch.zeros(
            (resolution * nx * resolution * ny), requires_grad=False
        ).to(device)
        for i in range(resolution * nx):
            for j in range(resolution * ny):
                xy[ctr, 0] = (i + 0.5) / resolution
                xy[ctr, 1] = (j + 0.5) / resolution
                if nonDesignRegion['Rect'] is not None:
                    if (
                        (xy[ctr, 0] < nonDesignRegion['Rect']['x<'])
                        and (xy[ctr, 0] > nonDesignRegion['Rect']['x>'])
                        and (xy[ctr, 1] < nonDesignRegion['Rect']['y<'])
                        and (xy[ctr, 1] > nonDesignRegion['Rect']['y>'])
                    ):
                        # nonDesignIdx.append(ctr);
                        nonDesignIdx[ctr] = 0
                if nonDesignRegion['Circ'] is not None:
                    if (
                        (xy[ctr, 0] - nonDesignRegion['Circ']['center'][0]) ** 2
                        + (xy[ctr, 1] - nonDesignRegion['Circ']['center'][1]) ** 2
                    ) <= nonDesignRegion['Circ']['rad'] ** 2:
                        # nonDesignIdx.append(ctr);
                        nonDesignIdx[ctr] = 0
                if nonDesignRegion['Annular'] is not None:
                    locn = (
                        xy[ctr, 0] - nonDesignRegion['Annular']['center'][0]
                    ) ** 2 + (xy[ctr, 1] - nonDesignRegion['Annular']['center'][1]) ** 2
                    if (locn <= nonDesignRegion['Annular']['rad_out'] ** 2) and (
                        locn > nonDesignRegion['Annular']['rad_in'] ** 2
                    ):
                        # nonDesignIdx.append(ctr);
                        nonDesignIdx[ctr] = 0
                ctr += 1
        xy = torch.tensor(xy, requires_grad=True).float().view(-1, 2).to(device)
        return xy, nonDesignIdx

    def initializeOptimizer(
        self,
        numLayers,
        numNeuronsPerLyr,
        desiredVolumeFraction,
        symXAxis=False,
        symYAxis=False,
    ):
        self.desiredVolumeFraction = desiredVolumeFraction
        self.density = self.desiredVolumeFraction * np.ones(
            (self.FE.nelx * self.FE.nely)
        )
        self.lossFunction = TopOptLoss()
        self.topNet = TopNet(
            numLayers, numNeuronsPerLyr, self.FE.nelx, self.FE.nely, symXAxis, symYAxis
        ).to(device)
        self.objective = 0.0
        self.convergenceHistory = []

        # NOTE: This is creating an empty plot because we have plotting turned
        # off
        # self.topFig, self.topAx = plt.subplots()
        # plt.ion()

    # %%
    def optimizeDesign(self, maxEpochs, minEpochs, useSavedNet):
        self.convergenceHistory = []
        savedNetFileName = (
            "./results/"
            + self.exampleName
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + '.nt'
        )
        learningRate = 0.01
        alphaMax = 100 * self.desiredVolumeFraction
        alphaIncrement = 0.05
        alpha = alphaIncrement
        # start
        nrmThreshold = 0.1
        # for gradient clipping
        if useSavedNet:
            if path.exists(savedNetFileName):
                self.topNet = torch.load(savedNetFileName)
            else:
                print("Network file not found")
        self.optimizer = optim.Adam(
            self.topNet.parameters(), amsgrad=True, lr=learningRate
        )

        for epoch in range(maxEpochs):
            batch_x = (
                torch.tensor(self.xy, requires_grad=True).view(-1, 2).float().to(device)
            )
            self.optimizer.zero_grad()
            nn_rho = torch.flatten(self.topNet(batch_x, self.nonDesignIdx)).to(device)

            # TODO: This is where we need to apply a mask for non-design regions

            rho_np = nn_rho.cpu().detach().numpy()
            # move tensor to numpy array
            u, Jelem = self.FE.solve88(rho_np)
            # Call FE 88 line code [Niels Aage 2013]
            if epoch == 0:
                self.obj0 = (self.FE.Emax * (rho_np**self.FE.penal) * Jelem).sum()
            # For sensitivity analysis, exponentiate by 2p here and divide by p in
            # the loss func hence getting -ve sign
            Jelem = (
                torch.tensor(self.FE.Emax * (rho_np ** (2 * self.FE.penal)) * Jelem)
                .view(-1)
                .float()
                .to(device)
            )
            objective, volConstraint = self.lossFunction(
                nn_rho, Jelem, self.desiredVolumeFraction, self.FE.penal, self.obj0
            )
            currentVolumeFraction = np.average(rho_np)
            self.objective = objective
            loss = self.objective + alpha * pow(volConstraint, 2)
            alpha = min(alphaMax, alpha + alphaIncrement)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            self.optimizer.step()
            if (
                volConstraint < 0.05
            ):  # Only check for gray when close to solving. Saves computational cost
                greyElements = sum(1 for rho in rho_np if ((rho > 0.05) & (rho < 0.95)))
                relGreyElements = greyElements / len(rho_np)
            else:
                relGreyElements = 1

            # Here we need to get the exact arguments for our constraints
            best_final_design = rho_np
            plotResolution = 1

            # The final design needs to be reshaped and transposed
            best_final_design = best_final_design.reshape(
                plotResolution * self.FE.nelx, plotResolution * self.FE.nely
            )
            best_final_design = best_final_design.T

            # Now compute our values
            binary_constraint = calculate_binary_constraint(
                design=best_final_design,
                epsilon=self.args['epsilon'],
            )
            volume_constraint = calculate_volume_constraint(
                design=best_final_design,
                volume=self.args['volfrac'],
            )

            self.convergenceHistory.append(
                [
                    self.objective.item(),
                    currentVolumeFraction,
                    loss.item(),
                    relGreyElements,
                    # We need to compare the unscaled compliance
                    self.objective.item() * self.obj0,
                    binary_constraint,
                    volume_constraint,
                ]
            )

            # Increasing penal after each epoch
            self.FE.penal = min(4.0, self.FE.penal + 0.01)

            # Plot the results
            # NOTE: For our pipeline we will turn this plotting off
            if epoch % 20 == 0:
                # self.plotTO(epoch)

                # Print training log
                print(
                    "{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} ".format(  # noqa
                        epoch,
                        self.objective.item() * self.obj0,
                        currentVolumeFraction,
                        loss.item(),
                        relGreyElements,
                    )
                )

            # Early stopping mechanism?
            # This is their criterion for early stopping - what happens if we
            # lower it?
            if (epoch > minEpochs) & (relGreyElements < 1e-5):
                break

        # NOTE: Turn off plotting for our pipeline
        # self.plotTO(epoch, True)

        # Note final logging
        print(
            "{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} ".format(
                epoch,
                self.objective.item() * self.obj0,
                currentVolumeFraction,
                loss.item(),
                relGreyElements,
            )
        )

        # NOTE: In our version to not save the data
        # torch.save(self.topNet, savedNetFileName)
        return self.convergenceHistory

    def plotTO(self, iter, saveFig=False):
        saveFrame = False
        # set this T/F if you want to create frames- use for video
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
        density = (
            torch.flatten(self.topNet(self.xyPlot, self.nonDesignPlotIdx))
            .detach()
            .cpu()
            .numpy()
        )
        a = plt.contourf(
            xx,
            yy,
            density.reshape(
                (
                    self.boundaryResolution * self.FE.nelx,
                    self.boundaryResolution * self.FE.nely,
                )
            ),
            2,
            cmap=plt.cm.jet,
        )
        self.topFig.colorbar(a)
        self.topFig.canvas.draw()
        plt.title(
            'Iter = {:d}, J = {:.2F}, V_f = {:.2F}, V_des = {:.2F}'.format(
                iter,
                self.objective * self.obj0,
                np.mean(density),
                self.desiredVolumeFraction,
            )
        )
        plt.axis('Equal')
        plt.grid(False)

        if saveFrame:
            plt.savefig('./frames/f_' + str(iter) + '.jpg')
        if saveFig:
            fig, ax = plt.subplots()

            # For us plot resolution is 1
            plotResolution = 1
            # controlslower quality figure
            xyPlot, nonDesignPlotIdx = self.generatePoints(
                self.FE.nelx, self.FE.nely, plotResolution, self.nonDesignRegion
            )
            Grids = True
            density = (
                torch.flatten(self.topNet(xyPlot, nonDesignPlotIdx))
                .detach()
                .cpu()
                .numpy()
            )

            if 0:  # plot BW contour
                xyPlot = xyPlot.cpu().detach().numpy()
                xx = np.reshape(
                    xyPlot[:, 0],
                    (plotResolution * self.FE.nelx, plotResolution * self.FE.nely),
                )
                yy = np.reshape(
                    xyPlot[:, 1],
                    (plotResolution * self.FE.nelx, plotResolution * self.FE.nely),
                )
                colr = plt.cm.binary
                a = plt.contourf(
                    xx,
                    yy,
                    density.reshape(
                        (plotResolution * self.FE.nelx, plotResolution * self.FE.nely)
                    ),
                    2,
                    cmap=colr,
                )
            else:  # plot pixel image
                ax.imshow(
                    -np.flipud(
                        density.reshape(
                            (
                                plotResolution * self.FE.nelx,
                                plotResolution * self.FE.nely,
                            )
                        ).T
                    ),
                    cmap='gray',
                    interpolation='none',
                    norm=colors.Normalize(vmin=-1, vmax=0),
                )
            if Grids:
                ax.xaxis.grid(True, zorder=0)
                ax.yaxis.grid(True, zorder=0)
                ax.set_xticks(
                    np.arange(0, plotResolution * self.FE.nelx + 1, plotResolution)
                )
                ax.set_yticks(
                    np.arange(0, plotResolution * self.FE.nely + 1, plotResolution)
                )
                ax.set_xticklabels(
                    np.array(ax.get_xticks().tolist()) / plotResolution,
                    fontsize=5,
                    rotation='vertical',
                )
                ax.set_yticklabels(
                    np.array(ax.get_yticks().tolist()) / plotResolution,
                    fontsize=5,
                    rotation='horizontal',
                )
                ax.axis('Equal')
                ax.grid(alpha=0.8)
                plt.grid(True)
            else:
                ax.axis('Equal')
                ax.axis('off')
            plt.title(
                'J = {:.2F}; iter = {:d}; V_f = {:.2F}'.format(
                    self.objective * self.obj0, iter, np.mean(density)
                ),
                y=-0.15,
                fontsize='xx-large',
            )
            fName = (
                "./results/"
                + self.exampleName
                + "_"
                + str(self.desiredVolumeFraction)
                + "_"
                + str(self.nelx)
                + "_"
                + str(self.nely)
                + '_topology.png'
            )
            fig.tight_layout()
            fig.savefig(fName, dpi=450)
            fig.show()
        plt.pause(0.01)

    def plotConvergence(self):
        self.convergenceHistory = np.array(self.convergenceHistory)
        plt.figure()
        plt.semilogy(self.convergenceHistory[:, 0], 'b:', label='Rel. Compliance')
        plt.semilogy(self.convergenceHistory[:, 1], 'r--', label='Vol. Fraction')
        # plt.yticks((0.01,0.035,0.1,0.25,0.5,1.0),("0.01","0.035","0.1","0.25","0.5","1.0"))
        plt.title('Convergence Plots')
        plt.title(
            'Convergence plots; V_des = {:.2F}'.format(self.desiredVolumeFraction)
        )
        plt.xlabel('Iterations')
        plt.grid('True')
        plt.legend(loc='lower left', shadow=True, fontsize='large')
        fName = (
            "./results/"
            + self.exampleName
            + "_"
            + str(self.desiredVolumeFraction)
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + '_convergence.png'
        )
        plt.savefig(fName, dpi=450)


# TODO: After pipeline is verified go back and clean this up
# #  ~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~#
# example = 6
# # see below for description
# #  ~~~~~~~~~~~~Main Simulation Parameters~~~~~~~~~~~~~#
# nelx = 60
# # number of FE elements along X
# nely = 30
# # number of FE elements along Y
# #  ~~~~~~~~~~~~Other Simulation Parameters~~~~~~~~~~~~~#
# numLayers = 5
# # the depth of the NN
# numNeuronsPerLyr = 20
# # the height of the NN
# minEpochs = 20
# # minimum number of iterations
# maxEpochs = 500
# # Max number of iterations
# penal = 2
# # SIMP penalization constant, starting value
# useSavedNet = False
# # use a net previouslySaved  as starting point
# # (exampleName_nelx_nely.nt in ./results folder)
# #  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
# # Tip Cantilever
# if example == 1:
#     exampleName = 'TipCantilever'
#     desiredVolumeFraction = 0.70
#     # between 0.1 and 0.9
#     ndof = 2 * (nelx + 1) * (nely + 1)
#     force = np.zeros((ndof, 1))
#     dofs = np.arange(ndof)
#     fixed = dofs[0 : 2 * (nely + 1) : 1]
#     force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
#     nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
#     symXAxis = False
#     symYAxis = False

# # Mid Cantilever
# elif example == 2:
#     exampleName = 'MidCantilever'
#     desiredVolumeFraction = 0.50
#     # between 0.1 and 0.9
#     ndof = 2 * (nelx + 1) * (nely + 1)
#     force = np.zeros((ndof, 1))
#     dofs = np.arange(ndof)
#     fixed = dofs[0 : 2 * (nely + 1) : 1]
#     force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
#     nonDesignRegion = {
#         'Rect': None,
#         'Circ': {'center': [30.0, 15.0], 'rad': 6.0},  # type: ignore
#         'Annular': None,
#     }
#     symXAxis = True
#     symYAxis = False

# # MBB Beam
# elif example == 3:
#     desiredVolumeFraction = 0.75
#     # between 0.1 and 0.9
#     exampleName = 'MBBBeam'
#     ndof = 2 * (nelx + 1) * (nely + 1)
#     force = np.zeros((ndof, 1))
#     dofs = np.arange(ndof)
#     fixed = np.union1d(
#         np.arange(0, 2 * (nely + 1), 2),
#         2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1,
#     )
#     force[2 * (nely + 1) + 1, 0] = -1
#     nonDesignRegion = {
#         'Rect': None,
#         'Circ': {'center': [30.0, 15.0], 'rad': 8},  # type: ignore
#         'Annular': None,
#     }
#     symXAxis = False
#     symYAxis = True

# # Michell
# elif example == 4:
#     desiredVolumeFraction = 0.34
#     # between 0.1 and 0.9
#     exampleName = 'Michell'
#     ndof = 2 * (nelx + 1) * (nely + 1)
#     force = np.zeros((ndof, 1))
#     dofs = np.arange(ndof)
#     fixed = np.array(
#         [
#             0,
#             1,
#             2 * (nelx + 1) * (nely + 1) - 2 * nely + 1,
#             2 * (nelx + 1) * (nely + 1) - 2 * nely,
#         ]
#     )
#     force[nelx * (nely + 1) + 1, 0] = -1
#     nonDesignRegion = {
#         'Rect': None,
#         'Circ': None,
#         'Annular': {'center': [30.0, 15.0], 'rad_out': 6.0, 'rad_in': 3},  # type: ignore  # noqa
#     }
#     symXAxis = False
#     symYAxis = True

# # Distributed MBB
# elif example == 5:
#     exampleName = 'Bridge'
#     desiredVolumeFraction = 0.5
#     # between 0.1 and 0.9
#     ndof = 2 * (nelx + 1) * (nely + 1)
#     force = np.zeros((ndof, 1))
#     dofs = np.arange(ndof)
#     fixed = np.array(
#         [
#             0,
#             1,
#             2 * (nelx + 1) * (nely + 1) - 2 * nely + 1,
#             2 * (nelx + 1) * (nely + 1) - 2 * nely,
#         ]
#     )
#     force[2 * nely + 1 : 2 * (nelx + 1) * (nely + 1) : 2 * (nely + 1), 0] = -1 / (
#         nelx + 1
#     )
#     nonDesignRegion = {
#         'Rect': {'x>': 0, 'x<': nelx, 'y>': nely - 1, 'y<': nely},  # type: ignore  # noqa
#         'Circ': None,
#         'Annular': None,
#     }
#     symXAxis = False
#     symYAxis = True

# # Tensile bar
# elif example == 6:
#     exampleName = 'TensileBar'
#     nelx = 20
#     # number of FE elements along X
#     nely = 10
#     # number of FE elements along Y
#     numLayers = 1
#     # the depth of the NN
#     numNeuronsPerLyr = 1
#     # the height of the NN
#     desiredVolumeFraction = 0.4
#     # between 0.1 and 0.9
#     ndof = 2 * (nelx + 1) * (nely + 1)
#     force = np.zeros((ndof, 1))
#     dofs = np.arange(ndof)
#     fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 1)
#     # fix X dof on left
#     midDofX = 2 * (nelx + 1) * (nely + 1) - (nely)
#     force[midDofX, 0] = 1
#     nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
#     symXAxis = True
#     symYAxis = False

# plt.close('all')
# start = time.perf_counter()
# topOpt = TopologyOptimizer()
# topOpt.initializeFE(exampleName, nelx, nely, force, fixed, penal, nonDesignRegion)
# topOpt.initializeOptimizer(
#     numLayers, numNeuronsPerLyr, desiredVolumeFraction, symXAxis, symYAxis
# )
# topOpt.optimizeDesign(maxEpochs, minEpochs, useSavedNet)
# print("Time taken (secs): {:.2F}".format(time.perf_counter() - start))
# topOpt.plotConvergence()
# modelWeights, modelBiases = topOpt.topNet.getWeights()
# print("#Design variables: ", len(modelWeights) + len(modelBiases))
