import cvxopt
import cvxopt.cholmod
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.sparse import coo_matrix

# -----------------------#


# %%  structural FE
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
        sK = ((self.KE.flatten()[np.newaxis]).T * (EnetOfElem)).flatten(order='F')
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
        im = plt.pcolormesh(x, y, z, cmap='jet')
        plt.title('strain X')
        fig.colorbar(im)

        plt.subplot(2, 2, 2)
        z = strainV.reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap='jet')
        plt.title('strain Y')
        fig.colorbar(im)

        plt.subplot(2, 2, 3)
        z = delta.reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap='jet')
        plt.title('net deformation')
        fig.colorbar(im)

        plt.subplot(2, 2, 4)
        z = (rhoElem * self.Jelem).reshape(self.nelx, self.nely)
        im = plt.pcolormesh(x, y, z, cmap='jet')
        plt.title('element compliance')
        fig.colorbar(im)

        # plt.subplot(2,2,2);
        # im = plt.imshow(np.fliplr(np.flipud(rhoElem*elemV).reshape((self.nelx,self.nely)).T), cmap=cm.jet,interpolation='none')  # noqa
        # fig.colorbar(im)
        # plt.title('|U_y|')

        # displacement = np.sqrt(elemU**2 + elemV**2);
        # plt.subplot(2,2,3);
        # im = plt.imshow(np.fliplr(np.flipud(rhoElem*displacement).reshape((self.nelx,self.nely)).T), cmap=cm.jet,interpolation='none')  # noqa
        # plt.title('||U||')
        # fig.colorbar(im)

        # plt.subplot(2,2,4);
        # im = plt.imshow(np.fliplr(np.flipud(rhoElem*self.Jelem).reshape((self.nelx,self.nely)).T), cmap=cm.jet,interpolation='none')  # noqa
        # fig.colorbar(im)
        # plt.title('Jelem')
        # fig.show()

        # xdef, ydef = x**2, y**2 + x

        fig, axes = plt.subplots(ncols=1)
        plt.pcolormesh(x, y, z, cmap='jet')


# %% test structural
def runFEA():
    plt.close('all')
    nelx = 60
    nely = 30
    penal = 3
    problem = 1
    ndof = 2 * (nelx + 1) * (nely + 1)
    f = np.zeros((ndof, 1))
    dofs = np.arange(ndof)

    if problem == 1:  # half MBB
        fixed = np.union1d(
            dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1])
        )
        f[1, 0] = -1
    elif problem == 2:  # cantilever
        fixed = dofs[0 : 2 * (nely + 1) : 1]
        f[2 * (nelx + 1) * (nely + 1) - (nely + 1) - 1, 0] = -1
    elif problem == 3:
        fixed = dofs[0 : 2 * (nely + 1) : 1]
        f[2 * (nelx + 1) * (nely + 1) - 1, 0] = -1
    elif problem == 4:
        numYnodes = nely + 1
        offset_botRght = 2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1)
        fixed = nely + np.array(
            [
                (nelx + 1) * (nely + 1),
                (nelx + 1) * (nely + 1) - 1,
                (nelx + 1) * (nely + 1) - (nely),
                (nelx + 1) * (nely + 1) - 1 - (nely),
            ]
        )
        for node in range(numYnodes):
            f[2 * node, 0] = -(node - nely * 0.5) / nely
            f[offset_botRght + 2 * node, 0] = (node - nely * 0.5) / nely

    EMaterials = np.array([1.0])
    density = np.ones(nely * nelx, dtype=float).reshape(-1, 1)
    fem = StructuralFE()
    fem.initializeSolver(nelx, nely, f, fixed, penal)
    fem.initializeMultiMaterial(EMaterials)
    u, J = fem.solve88(density)
    fem.plotFE()
    return fem


# fem = runFEA(); # test FEA . Comment/uncomment this to run
