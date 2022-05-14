import numpy as np
from dmrg_finite import DMRG, solve_E0, Sz_value,truncate_theta,entropy
from scipy.linalg import expm
import matplotlib.pyplot as plt


def doHbonds(L,J,h,dt):
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)
    H_bonds = []
    for i in range(L - 1):
        hL = hR = 0.5 * h
        if i == 0:
            hL = h
        if i + 1 == L - 1:
            hR = h
        H_bond = -J * np.kron(sx, sx) - hL * np.kron(sz, id) - hR * np.kron(id, sz)
        H = np.reshape(H_bond, [2 * 2, 2 * 2])
        U = expm(-dt * H)
        H_bonds.append(np.reshape(U, [2, 2, 2, 2]))
    return H_bonds


def update_bond(Bs,Ss,i,H,chi,eps):
    j=i+1
    x1=np.tensordot(np.diag(Ss[i]),Bs[i],[1,0])
    theta=np.tensordot(x1,Bs[j],[2,0])
    Utheta=np.tensordot(H,theta,axes=([2,3],[1,2]))
    Utheta=np.transpose(Utheta,[2,0,1,3])

    Ai,Sj,Bj=truncate_theta(Utheta,chi,eps)
    Gi=np.tensordot(np.diag(Ss[i]**(-1)),Ai,axes=[1,0])
    Bs[i]=np.tensordot(Gi,np.diag(Sj),axes=[2,0])
    Ss[j]=Sj
    Bs[j]=Bj

    return Bs,Ss


def TEBD(L,J,h,Bs,Ss,dt,chi,eps):
    H_bonds=doHbonds(L,J,h,dt)
    for k in [0,1]:
        for i in range(k,L-1,2):
            Bs,Ss=update_bond(Bs,Ss,i,H_bonds[i],chi,eps)
    return Bs,Ss

if __name__ == "__main__":
    L = 50; J = 1.;  h1 = 0.3;  h2 =0.5
    chi = 50;  sweeps = 30;
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)

    #initialize MPS
    B = np.zeros([1, 2, 1], dtype=float)
    B[0, 0, 0] = 1.
    S = np.ones([1], dtype=float)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]

    #initialize MPO
    mpo = []
    for i in range(L):
        w = np.zeros((3, 3, 2, 2), dtype=float)
        w[0, 0] = w[2, 2] = id
        w[0, 1] = sx
        w[0, 2] = -h1 * sz
        w[1, 2] = -J * sx
        mpo.append(w)

    #find ground state
    Bs, Ss = DMRG(L, Bs, Ss, mpo, chi, sweeps)
    sigma_zs=[]; energy=[]; S=[]
    print("TEBD_finite " "L=", L, " h=", h1)
    sigma_z = np.sum(Sz_value(L, Bs, Ss))
    E0 = np.sum(solve_E0(L, Bs, Ss, J, h1))
    ee=entropy(Ss,int(L/2))
    S.append(ee)
    sigma_zs.append(sigma_z)
    energy.append(E0)
    print("t={t:f}, E0={E0:.5f},<sigma_z>={sigmaz:.5f}".format(t=0, E0=E0 / L,sigmaz=sigma_z))

    #tebd real time evolution
    tmax=30; dt=0.01;
    eps=1.e-10
    Nsteps=int(tmax/dt)
    for i in range(Nsteps):
        Bs,Ss=TEBD(L,J,h2,Bs,Ss,1.j*dt,chi,eps)
        sigma_z = np.sum(Sz_value(L, Bs, Ss))
        E0 = np.sum(solve_E0(L, Bs, Ss, J, h2))
        ee=entropy(Ss,int(L/2))
        S.append(ee)
        sigma_zs.append(sigma_z)
        energy.append(E0)
        print("t=", (i + 1) * dt, "   entangle=", ee)
        #print("t={t:f}, E0={E0:.5f},<sigma_z>={sigmaz:.8f}".format(t=(i+1)*dt, E0=E0 / L,sigmaz=sigma_z))

    time=np.linspace(0,tmax,Nsteps+1)
    fig=plt.figure()
    plt.subplot(2,1,1)

    plt.plot(time,sigma_zs)
    plt.subplot(2,1,2)

    plt.plot(time,S)
    plt.grid()
    plt.show()





