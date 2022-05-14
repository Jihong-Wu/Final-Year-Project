import numpy as np
import matplotlib.pyplot as plt
from dmrg_finite import DMRG, solve_E0, Sz_value


def magnetization_value2(L,Bs,Ss):
    result=[]
    sz=np.array([[1.,0.],[0.,-1.]])
    c=L//2
    mz=np.tensordot


if __name__ == "__main__":
    L = 20;
    J = 0.5;
    h = 0.1;
    chi = 30;
    sweeps = 10;
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)

    # initialize MPS
    B = np.zeros([1, 2, 1], dtype=float)
    B[0, 0, 0] = 1.
    S = np.ones([1], dtype=float)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]

    # initialize MPO
    mpo = []
    for i in range(L):
        w = np.zeros((3, 3, 2, 2), dtype=float)
        w[0, 0] = w[2, 2] = id
        w[0, 1] = sx
        w[0, 2] = -h * sz
        w[1, 2] = -J * sx
        mpo.append(w)

    Bs, Ss = DMRG(L, J, h, Bs, Ss, mpo, chi, sweeps)
    # calculations
    print("DMRG_finite", "L=", L, " h=", h)
    sigma_z = np.sum(Sz_value(L, Bs, Ss))
    print("<sigma_z>=", sigma_z)
    E0 = np.sum(solve_E0(L, Bs, Ss, J, h))
    print("E0=", E0 / L)