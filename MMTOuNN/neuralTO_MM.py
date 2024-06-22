'''
MMTOuNN : Multi-Material Topology Optimization using Neural Networks
Aaditya Chandrasekhar, Krishnan Suresh
Submitted to Structural and Multidisciplinary Optimization, 2020
achandrasek3@wisc.edu
ksuresh@wisc.edu
For academic purposes only
'''

import random
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import colors

from MMTOuNN.FE_MM import StructuralFE

plt.rcParams['figure.dpi'] = 150

# %%  set device CPU/GPU
devices = {'cpu': 0, 'gpu': 1}


def setDevice(device):
    if torch.cuda.is_available() and (devices[device] == 1):
        device = torch.device("cuda:0")
        print("GPU enabled")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device


# %% seed
def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)


# %% init NN
class TopNet(nn.Module):
    def __init__(
        self,
        numLayers,
        numNeuronsPerLyr,
        nelx,
        nely,
        numMaterials,
        symXAxis,
        symYAxis,
        seed,
    ):
        self.inputDim = 2
        # x and y coordn of the point
        self.outputDim = numMaterials + 1
        # if material A/B/.../void at the point
        self.nelx = nelx
        self.nely = nely
        self.symXAxis = symXAxis
        self.symYAxis = symYAxis
        self.numLayers = numLayers
        self.numNeuronsPerLyr = numNeuronsPerLyr
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = self.inputDim
        set_seed(seed)
        # NN are seeded manually

        for lyr in range(numLayers):
            l = nn.Linear(current_dim, numNeuronsPerLyr)  # noqa
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = numNeuronsPerLyr

        self.layers.append(nn.Linear(current_dim, self.outputDim))
        self.bnLayer = nn.ModuleList()
        for lyr in range(numLayers):
            self.bnLayer.append(nn.BatchNorm1d(numNeuronsPerLyr))

    def forward(self, x, fixedIdx=None):
        # activations ReLU, ReLU6, ELU, SELU, PReLU, LeakyReLU, Sigmoid,
        # Tanh, LogSigmoid, Softplus, Softsign,
        # TanhShrink, Softmin, Softmax
        m = nn.ReLU6()
        #
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
        for layer in self.layers[:-1]:
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        out = 1e-4 + torch.softmax(self.layers[-1](x), dim=1)

        if fixedIdx is not None:
            out[fixedIdx, 0] = 0.95
            out[fixedIdx, 1:] = 0.01
            # fixed Idx removes region
        return out

    def getWeights(self):  # stats about the NN
        modelWeights = []
        modelBiases = []
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.cpu().view(-1).numpy())
            modelBiases.extend(lyr.bias.data.cpu().view(-1).numpy())
        return modelWeights, modelBiases


# %% loss function
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
    # %% topoOpt
    def initializeFE(
        self,
        problemName,
        nelx,
        nely,
        elemArea,
        forceBC,
        fixed,
        device='cpu',
        penal=3,
        nonDesignRegion=None,
        EMaterials=np.array([1.0]),
    ):
        self.problemName = problemName
        self.nelx = nelx
        self.nely = nely
        self.boundaryResolution = 2
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
                        (xy[ctr, 0] < nonDesignRegion['x<'])
                        and (xy[ctr, 0] > nonDesignRegion['x>'])
                        and (xy[ctr, 1] < nonDesignRegion['y<'])
                        and (xy[ctr, 1] > nonDesignRegion['y>'])
                    ):
                        nonDesignIdx.append(ctr)
                ctr += 1
        xy = torch.tensor(xy, requires_grad=True).float().view(-1, 2).to(self.device)
        return xy, nonDesignIdx

    def initializeOptimizer(
        self,
        numLayers,
        numNeuronsPerLyr,
        desiredMassFraction,
        massDensityMaterials,
        symXAxis=False,
        symYAxis=False,
        seed=1234,
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
        self.topNet = TopNet(
            numLayers,
            numNeuronsPerLyr,
            self.FE.nelx,
            self.FE.nely,
            self.FE.numMaterials,
            symXAxis,
            symYAxis,
            seed=seed,
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

    # %% optimize design
    def optimizeDesign(self, maxEpochs, minEpochs, useSavedNet):
        lossHistory = []
        savedNetFileName = (
            "./results/"
            + self.problemName
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + '.nt'
        )
        if useSavedNet:
            if path.exists(savedNetFileName):
                self.topNet = torch.load(savedNetFileName)
            else:
                print("Network file not found")
        self.optimizer = optim.Adam(
            self.topNet.parameters(), amsgrad=True, lr=0.01, weight_decay=1e-5
        )
        objHist = []
        alpha = 0.5
        alphaIncrement = 0.15
        nrmThreshold = 0.2
        self.numIter = 0
        batch_x = self.xy.to(self.device)

        for epoch in range(maxEpochs):
            self.optimizer.zero_grad()
            nnPred = self.topNet(batch_x, self.nonDesignIdx)[:, 1:].to(self.device)
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
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            self.optimizer.step()
            self.massConstraint = massConstraint.detach().cpu().numpy()
            objHist.append(loss.item())
            modelWeights, modelBiases = self.topNet.getWeights()
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

            # Update the penalty that starts at 1 for each epoch
            self.FE.penal = min(4.0, self.FE.penal + 0.01)
            # continuation scheme
            lossHistory.append(
                [
                    self.massConstraint,
                    self.objective.item(),
                    loss.item(),
                    [np.mean(nnPred_np[:, i]) for i in range(self.FE.numMaterials)],
                    # We need to compare the unscaled compliance
                    self.objective.item() * self.obj0,
                ]
            )
            self.numIter = self.numIter + 1
            alpha = min(100, alpha + alphaIncrement)
            # if epoch % 24 == 0:
            #     self.plotMaterialContour(epoch)
            #     print(
            #         "{:d} p {:.3F} J0 {:.3F};  loss {:.3F}; massCons {:.3F}; relGray {:.3F}".format(  # noqa
            #             epoch,
            #             self.FE.penal,
            #             self.objective.item(),
            #             loss.item(),
            #             self.massConstraint,
            #             relGreyElements,
            #         )
            #     )
            if (
                (epoch > minEpochs)
                and (np.abs(self.massConstraint) < 0.05)
                and (relGreyElements < 0.035)
            ):
                # print(
                #     "{:d} p {:.3F} J0 {:.3F};  loss {:.3F}; massCons {:.3F}; relGray {:.3F}".format(  # noqa
                #         epoch,
                #         self.FE.penal,
                #         self.objective.item(),
                #         loss.item(),
                #         self.massConstraint,
                #         relGreyElements,
                #     )
                # )
                break
        # self.FE.plotFE()
        torch.save(self.topNet, savedNetFileName)
        return lossHistory

    # %% plots

    def plotMaterialContour(
        self,
        iter,
        saveFig=False,
        fillColors=['0.90', 'red', 'cyan', 'black', 'pink', 'blue'],
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
        nnPred = self.topNet(self.xyPlot, self.nonDesignPlotIdx).to(self.device)
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
            'Iter = {:d}, J = {:.2F}, m/m*= {:.2F}'.format(
                iter, self.objective * self.obj0, self.massConstraint + 1
            )
        )
        plt.axis('Equal')
        plt.pause(0.0001)
        if saveFig or iter % 24 == 0:
            plt.savefig('./frames/material_f_' + str(iter) + '.jpg')

    def plotMaterialImage(
        self,
        resolution=5,
        grids=False,
        fillColors=['0.90', 'red', 'cyan', 'black', 'pink', 'blue'],
    ):
        xyPlot, nonDesignPlotIdx = self.generatePoints(
            self.FE.nelx, self.FE.nely, resolution, self.nonDesignRegion
        )
        nnPred_np = self.topNet(xyPlot, nonDesignPlotIdx).detach().cpu().numpy()
        matIdx = np.array([np.argmax(rw) for rw in nnPred_np])
        fig, ax = plt.subplots()
        plt.imshow(
            np.flipud(
                (matIdx)
                .reshape((resolution * self.FE.nelx, resolution * self.FE.nely))
                .T
            ),
            cmap=colors.ListedColormap(fillColors),
            interpolation='none',
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
                rotation='vertical',
            )
            ax.set_yticklabels(
                np.array(ax.get_yticks().tolist()) / resolution,
                fontsize=5,
                rotation='horizontal',
            )
            ax.axis('Equal')
            ax.grid(alpha=0.8)
            plt.grid(True)
        else:
            ax.axis('Equal')
            ax.axis('off')
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=clr) for clr in fillColors]
        plt.legend(
            proxy,
            np.arange(0, self.FE.numMaterials + 1),
            ncol=min(4, self.FE.numMaterials + 1),
        )

        netMass = (1 + self.massConstraint) * self.desiredMass
        plt.title(
            'J = {:.2F}; m = {:.2E}; iter = {:d} '.format(
                self.objective * self.obj0, netMass, self.numIter
            ),
            y=-0.15,
            fontsize='xx-large',
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
            + '_topology.png'
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
            plt.title('Density gradient of material {:d}'.format(i))
            fig.show()
            if saveFig or iter % 24 == 0:
                plt.savefig('./results/densityGrad_' + str(i) + '.jpg')

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
        self.topAx.axis('equal')
        self.topFig.canvas.draw()
        plt.title(
            'Iter = {:d}, J = {:.2F}, m/m*= {:.2F}'.format(
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
                + '_convergence.png'
            )
            plt.savefig(fName, dpi=450)

    def plotConvergence(self):
        self.convergenceHistory = np.array(self.convergenceHistory)
        plt.figure()
        plt.plot(self.convergenceHistory[:, 1], 'b:', label='Rel.Compliance')
        plt.plot(self.convergenceHistory[:, 0], 'r--', label='Mass constraint')
        plt.title(
            'Convergence Plots '
            + self.problemName
            + str(' m* = ')
            + str(self.desiredMassFraction)
        )
        plt.xlabel('Iterations')
        # plt.grid('on')
        plt.legend(loc='upper right', shadow=True, fontsize='large')
        fName = (
            "./results/"
            + self.problemName
            + "_"
            + str(self.desiredMass)
            + "_"
            + str(self.nelx)
            + "_"
            + str(self.nely)
            + '_convergence.png'
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


# %% Example - See paperExamples for more
def runExample():
    nelx = 64
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
    exampleName = 'TipCantilever'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = dofs[0 : 2 * (nely + 1) : 1]
    force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = False

    #  ~~~~~~~~~~~~Run code~~~~~~~~~~~~~#
    plt.close('all')
    minEpochs = 50
    maxEpochs = 500
    penal = 1.0
    useSavedNet = False
    device = 'cpu'
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
        desiredMassFraction,
        massDensityMaterials,
        symXAxis,
        symYAxis,
    )
    # topOpt.setDesiredMass(450);
    _ = topOpt.train(maxEpochs, minEpochs, useSavedNet)
    print("Time taken: {:.2F}".format(time.perf_counter() - start))
    modelWeights, modelBiases = topOpt.topNet.getWeights()

    #  ~~~~~~~~~~~Post process~~~~~~~~~~~~~#
    resolution = 15
    fillColors = ['0.90', 'red', 'cyan', 'black']
    topOpt.plotMaterialImage(resolution, False, fillColors)


# runExample(); # Comment to run headless
