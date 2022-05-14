import numpy as np
import matplotlib.pyplot as plt
from dmrg_infinite import DMRG


def solve_E1(L,Bs,Ss,J,h):
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id=np.eye(2)
    H_bonds=[]
    for i in range(L):
        hL=hR=0.5*h
        H_bond=-J*(np.kron(sx,sx)+np.kron(sy,sy))
        H_bonds.append(np.reshape(H_bond,[2,2,2,2]))
    result=[]
    for i in range(L):
        x1=np.tensordot(np.diag(Ss[i]),Bs[i],[1,0])
        j=(i+1)%2
        theta=np.tensordot(x1,Bs[j],[2,0])
        op_theta=np.tensordot(H_bonds[i],theta,axes=[[2,3],[1,2]])
        result.append(np.tensordot(theta.conj(),op_theta,[[0,1,2,3],[2,0,1,3]]))
    return np.real_if_close(result)



if __name__ == "__main__":
    L = 2;
    chi = 30;
    sweeps = 60;
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
        w = np.zeros((4, 4, 2, 2), dtype=complex)
        w[0, 0] = w[3, 3] = id
        w[1, 3] = w[0, 1] = sx
        w[2, 3] = w[0, 2] = sy
        mpo.append(w)

    Bs, Ss = DMRG(L, Bs, Ss, mpo, chi, sweeps)
    # calculations
    print("DMRG_finite", "Heisenberg_XX")
    E0 = np.mean(solve_E1(L, Bs, Ss, 1/4, 0))
    print("E0=", E0 )
    E0_exact=-1/np.pi
    print("E0_exact=",E0_exact)
    print("erro=",(-E0-E0_exact)/E0_exact*100)