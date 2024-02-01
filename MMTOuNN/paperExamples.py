# stdlib
import time

# third party
import matplotlib.pyplot as plt
import numpy as np
from neuralTO_MM import TopologyOptimizer

example = 7
plt.close('all')

if example == 1:  # Validation examples
    minEpochs = 50
    maxEpochs = 500
    penal = 1.0
    useSavedNet = False
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN

    nelx = 60
    nely = 30
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = 0.5

    exampleName = 'MBBBeam'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = np.union1d(
        np.arange(0, 2 * (nely + 1), 2),
        2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1,
    )
    force[2 * (nely + 1) + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = False

    # [1,2,3,4]
    EMaterials = np.array([2, 1.0, 0.6, 0.01])
    # Young modulus of the materials # , 2.0
    massDensityMaterials = np.array([1.0, 0.6, 0.4, 0.1])
    # wt (/area) of the candidate materials
    fillColors = ['0.90', 'black', 'red', 'blue', 'pink']

    # [1,3,4]
    # EMaterials =  np.array([2,  0.6, 0.01]); # Young modulus of the materials # , 2.0
    # massDensityMaterials = np.array([1., 0.4, 0.1]); # wt (/area) of the candidate materials  # noqa
    # fillColors = ['0.90','black','blue','pink'];

    # [1,2,4]]
    # EMaterials =  np.array([2,  1.0, 0.01]); # Young modulus of the materials # , 2.0
    # massDensityMaterials = np.array([1., 0.6, 0.1]); # wt (/area
    # fillColors = ['0.90','black','red','pink'];

    # [2,3,4]
    # EMaterials =  np.array([1.0, 0.6, 0.01]); # Young modulus of the materials # , 2.0
    # massDensityMaterials = np.array([0.6, 0.4, 0.1]); # wt (/area
    # fillColors = ['0.90','red','blue','pink'];

    # [1,4]
    # EMaterials =  np.array([2.0, 0.01]); # Young modulus of the materials # , 2.0
    # massDensityMaterials = np.array([1.0, 0.1]); # wt (/area
    # fillColors = ['0.90','black','pink'];

    # [2,4]
    # EMaterials =  np.array([1.0, 0.01]); # Young modulus of the materials # , 2.0
    # massDensityMaterials = np.array([0.6, 0.1]); # wt (/area
    # fillColors = ['0.90','red','pink'];

    start = time.perf_counter()
    topOpt = TopologyOptimizer()
    topOpt.initializeFE(
        exampleName,
        nelx,
        nely,
        elemArea,
        force,
        fixed,
        'cpu',
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
    topOpt.setDesiredMass(900)
    lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
    print("Time taken: {:.2F}".format(time.perf_counter() - start))
    topOpt.plotMaterialImage(resolution=15, grids=False, fillColors=fillColors)

    modelWeights, modelBiases = topOpt.topNet.getWeights()

if example == 2:  # Validity michell
    minEpochs = 50
    maxEpochs = 500
    penal = 2.0
    useSavedNet = False
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN

    nelx = 60
    nely = 30
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = 0.4

    # [1,2,3]
    EMaterials = np.array([1.0, 0.6, 0.2])
    massDensityMaterials = np.array([1.0, 0.7, 0.4])
    #
    fillColors = ['0.90', 'black', 'red', 'blue']

    # [1,2]
    EMaterials = np.array([1.0, 0.6])
    massDensityMaterials = np.array([1.0, 0.7])
    #
    fillColors = ['0.90', 'black', 'red']

    # [1,3]
    EMaterials = np.array([1.0, 0.2])
    massDensityMaterials = np.array([1.0, 0.4])
    #
    fillColors = ['0.90', 'black', 'blue']

    # [1]
    EMaterials = np.array([1.0])
    # Young modulus of the materials # , 2.0
    massDensityMaterials = np.array([1.0])
    #
    fillColors = ['0.90', 'black']

    exampleName = 'Michell'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1])
    force[int(2 * nelx * (nely + 1) / 4) + 1, 0] = -1
    # nelx should be multiple of 4
    force[int(2 * nelx * (nely + 1) / 2) + 1, 0] = -2
    force[int(2 * nelx * (nely + 1) * 3 / 4) + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = True

    start = time.perf_counter()
    topOpt = TopologyOptimizer()
    topOpt.initializeFE(
        exampleName,
        nelx,
        nely,
        elemArea,
        force,
        fixed,
        'cpu',
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

    lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
    print("Time taken: {:.2F}".format(time.perf_counter() - start))

    topOpt.plotMaterialImage(resolution=15, grids=False, fillColors=fillColors)
    modelWeights, modelBiases = topOpt.topNet.getWeights()

if example == 3:  # convergecne
    minEpochs = 50
    maxEpochs = 500
    penal = 1.0
    useSavedNet = False
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN

    nelx = 60
    nely = 30
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = 0.5
    EMaterials = np.array([380, 210, 110])
    # Young modulus of the materials # , 2.0 np.array([3.]) #
    massDensityMaterials = np.array([19250, 7800, 4390])
    # wt (/area np.array([1.]); #

    fillColors = ['0.90', 'red', 'cyan', 'black']

    exampleName = 'MidCantilever'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = dofs[0 : 2 * (nely + 1) : 1]
    force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
    nonDesignRegion = None
    symXAxis = True
    symYAxis = False

    device = 'gpu'
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
    lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
    print("Time taken: {:.2F}".format(time.perf_counter() - start))
    topOpt.plotConvergence()
    topOpt.plotMaterialImage(resolution=15, grids=True, fillColors=fillColors)

if example == 4:  # comp cost michell
    minEpochs = 50
    maxEpochs = 500
    penal = 2.0
    useSavedNet = False
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN
    elemArea = 1.0

    nelx = 60
    nely = 30

    desiredMassFraction = 0.7  # [1., 0.9, 0.7, 0.5, 0.3];

    maxNumMat = 11
    fillColors = [
        '0.90',
        'red',
        'cyan',
        'black',
        'orange',
        'blue',
        'orange',
        'green',
        'yellow',
        'magenta',
        'navy',
    ]

    exampleName = 'Michell'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1])
    force[int(2 * nelx * (nely + 1) / 4) + 1, 0] = -1
    # nelx should be multiple of 4
    force[int(2 * nelx * (nely + 1) / 2) + 1, 0] = -2
    force[int(2 * nelx * (nely + 1) * 3 / 4) + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = True
    devices = ['cpu', 'gpu']

    for numMat in range(1, maxNumMat):
        for dvc in devices:
            EMaterials = np.flip(np.linspace(2, 2 - 0.1 * numMat, numMat))
            massDensityMaterials = 0.5 * EMaterials

            start = time.perf_counter()
            topOpt = TopologyOptimizer()
            topOpt.initializeFE(
                exampleName,
                nelx,
                nely,
                elemArea,
                force,
                fixed,
                dvc,
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
            lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
            print("Time taken: {:.2F}".format(time.perf_counter() - start))

if example == 5:  # Neural net arch dependency
    minEpochs = 100
    maxEpochs = 500
    penal = 2.0
    useSavedNet = False
    numLayers = [4, 5, 7, 8, 9, 10]  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = [12, 15, 25, 30, 35, 50]  # type: ignore
    # the height of the NN

    nelx = 60
    nely = 30
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = 0.5
    EMaterials = np.array([380, 210, 110])
    massDensityMaterials = np.array([19250, 7800, 4390])
    fillColors = ['0.90', 'red', 'cyan', 'black', 'pink', 'blue']

    exampleName = 'TipCantilever'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = dofs[0 : 2 * (nely + 1) : 1]
    force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = False

    for i in range(len(numLayers)):  # type: ignore
        start = time.perf_counter()
        topOpt = TopologyOptimizer()
        topOpt.initializeFE(
            exampleName,
            nelx,
            nely,
            elemArea,
            force,
            fixed,
            'gpu',
            penal,
            nonDesignRegion,
            EMaterials,
        )
        topOpt.initializeOptimizer(
            numLayers[i],  # type: ignore
            numNeuronsPerLyr[i],  # type: ignore
            desiredMassFraction,
            massDensityMaterials,
            symXAxis,
            symYAxis,
        )
        lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
        print("Time taken: {:.2F}".format(time.perf_counter() - start))
        topOpt.plotMaterialImage(
            resolution=15, grids=False, fillColors=['0.90', 'red', 'cyan', 'black']
        )
        modelWeights, modelBiases = topOpt.topNet.getWeights()

if example == 6:  # mesh dependence michell
    minEpochs = 50
    maxEpochs = 500
    penal = 2.0
    useSavedNet = False
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN
    lengthX = 20
    lengthY = 10

    nelx = [24, 40, 60, 100]  # type: ignore
    nely = [12, 20, 30, 50]  # type: ignore

    desiredMassFraction = 0.5

    EMaterials = np.array([380, 210, 110])
    massDensityMaterials = np.array([19250, 7800, 4390])
    fillColors = ['0.90', 'red', 'cyan', 'black']

    exampleName = 'Michell'
    for i in range(len(nelx)):  # type: ignore
        ndof = 2 * (nelx[i] + 1) * (nely[i] + 1)  # type: ignore
        elemArea = lengthX * lengthY / (nelx[i] * nely[i])  # type: ignore
        force = np.zeros((ndof, 1))
        dofs = np.arange(ndof)
        fixed = np.array([0, 1, 2 * (nelx[i] + 1) * (nely[i] + 1) - 2 * nely[i] + 1])  # type: ignore  # noqa
        force[int(2 * nelx[i] * (nely[i] + 1) / 4) + 1, 0] = -1  # type: ignore
        # nelx should be multiple of 4
        force[int(2 * nelx[i] * (nely[i] + 1) / 2) + 1, 0] = -2  # type: ignore
        force[int(2 * nelx[i] * (nely[i] + 1) * 3 / 4) + 1, 0] = -1  # type: ignore
        nonDesignRegion = None
        symXAxis = False
        symYAxis = True

        start = time.perf_counter()
        topOpt = TopologyOptimizer()
        topOpt.initializeFE(
            exampleName,
            nelx[i],  # type: ignore
            nely[i],  # type: ignore
            elemArea,
            force,
            fixed,
            'cpu',
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
        lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
        print("Time taken: {:.2F}".format(time.perf_counter() - start))
        topOpt.plotConvergence()
        topOpt.plotMaterialImage(topOpt.numIter, True, fillColors)
        modelWeights, modelBiases = topOpt.topNet.getWeights()
        # topOpt.plotDensityGradient();
        # topOpt.plotMaterialImage(resolution =15, grids = True, fillColors = fillColors);  # noqa

if example == 7:  # Pareto w and w/o restart
    minEpochs = 50
    maxEpochs = 500
    penal = 1.0
    useSavedNet = False
    Restart = True
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN

    nelx = 60
    nely = 30
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2]  # type: ignore
    EMaterials = np.array([3, 2, 1])
    # Young modulus of the materials # , 2.0 np.array([3.]) #
    massDensityMaterials = np.array([1, 0.7, 0.4])
    # wt (/area np.array([1.]); #

    fillColors = ['0.90', 'red', 'cyan', 'black', 'pink', 'blue']

    exampleName = 'MidCantilever'
    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = dofs[0 : 2 * (nely + 1) : 1]
    force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
    nonDesignRegion = None
    symXAxis = True
    symYAxis = False

    for mfrac in desiredMassFraction:  # type: ignore
        device = 'cpu'
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
            numLayers, numNeuronsPerLyr, mfrac, massDensityMaterials, symXAxis, symYAxis
        )
        lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
        print("Time taken: {:.2F}".format(time.perf_counter() - start))
        # topOpt.plotConvergence();
        topOpt.plotMaterialContour(topOpt.numIter, False, fillColors)
        if not Restart:
            useSavedNet = True

if example == 8:  # Boundary extraction
    minEpochs = 50
    maxEpochs = 500
    penal = 2.0
    useSavedNet = False
    numLayers = 5  # type: ignore
    # the depth of the NN
    numNeuronsPerLyr = 25
    # the height of the NN

    nelx = 60
    nely = 30
    elemArea = 1.0
    # assumes sizeX = sizeY.
    desiredMassFraction = 0.3  # [1., 0.9, 0.7, 0.5, 0.3];
    EMaterials = np.array([380, 210, 110])
    # Young modulus of the materials # , 2.0
    massDensityMaterials = np.array([19250, 7800, 4390])
    # wt (/area
    fillColors = ['0.90', 'red', 'cyan', 'black']
    exampleName = 'Michell'
    ndof = 2 * (nelx + 1) * (nely + 1)
    elemArea = 1.0
    force = np.zeros((ndof, 1))
    dofs = np.arange(ndof)
    fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1])
    force[int(2 * nelx * (nely + 1) / 4) + 1, 0] = -1
    # nelx should be multiple of 4
    force[int(2 * nelx * (nely + 1) / 2) + 1, 0] = -2
    force[int(2 * nelx * (nely + 1) * 3 / 4) + 1, 0] = -1
    nonDesignRegion = None
    symXAxis = False
    symYAxis = True

    start = time.perf_counter()
    topOpt = TopologyOptimizer()
    topOpt.initializeFE(
        exampleName,
        nelx,
        nely,
        elemArea,
        force,
        fixed,
        'gpu',
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
    lossHist = topOpt.train(maxEpochs, minEpochs, useSavedNet)
    print("Time taken: {:.2F}".format(time.perf_counter() - start))
    topOpt.plotDensityGradient(saveFig=True)
    topOpt.plotMaterialImage(resolution=1, grids=True, fillColors=fillColors)
    modelWeights, modelBiases = topOpt.topNet.getWeights()
