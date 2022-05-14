import numpy as np
from dmrg_infinite import DMRG, solve_E0, Sz_value,truncate_theta
from dmrg_finite import entropy
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math


def doHbonds(L,J,h,dt):
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)
    H_bonds = []
    for i in range(L):
        hL = hR = 0.5 * h
        H_bond = -J * np.kron(sx, sx) - hL * np.kron(sz, id) - hR * np.kron(id, sz)
        H = np.reshape(H_bond, [2 * 2, 2 * 2])
        U = expm(-dt * H)
        H_bonds.append(np.reshape(U, [2, 2, 2, 2]))
    return H_bonds


def update_bond(Bs,Ss,i,H,chi,eps):
    j=(i+1)%L
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
        Bs,Ss=update_bond(Bs,Ss,k,H_bonds[k],chi,eps)
    return Bs,Ss

if __name__ == "__main__":
    L = 2; J = 1.;  h1 = 1.5;  h2 =0.5
    chi = 100;  sweeps = 50;
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
    sigma_zs = [];  energy = []; S=[]
    Bs, Ss = DMRG(L, Bs, Ss, mpo, chi, sweeps)
    #print("TEBD_infinite " "L=", L, " h=", h1)
    sigma_z = np.mean(Sz_value(L, Bs, Ss))
    #E0 = np.mean(solve_E0(L, Bs, Ss, J, h1))
    ee=entropy(Ss,1)
    sigma_zs.append(sigma_z)
    #energy.append(E0)
    S.append(ee)
    #print("t={t:f}, E0={E0:.5f},<sigma_z>={sigmaz:.5f}".format(t=0, E0=E0,sigmaz=sigma_z))

    #tebd real time evolution
    tmax=50; dt=0.01;
    eps=1.e-10
    Nsteps=int(tmax/dt)
    for i in range(Nsteps):
        Bs,Ss=TEBD(L,J,h2,Bs,Ss,1.j*dt,chi,eps)
        sigma_z = np.sum(Sz_value(L, Bs, Ss))
        #E0 = np.mean(solve_E0(L, Bs, Ss, J, h2))
        ee=entropy(Ss,0)
        sigma_zs.append(sigma_z)
        #energy.append(E0)
        S.append(ee)
        print("t=", (i + 1) * dt, "   entangle=", ee)
        #print("t={t:f}, E0={E0:.5f},<sigma_z>={sigmaz:.5f}".format(t=(i+1)*dt, E0=E0,sigmaz=sigma_z))
    time=np.linspace(0,tmax,Nsteps+1)



#exact
    q = np.linspace(0., np.pi, 320)
    sq = np.sin(q)
    Nq = len(q)
    t = np.linspace(0, tmax, 200)
    Nt = len(t)
    h_0 = h1
    h_1 = h2
    epsi_0 = h_0 - np.cos(q)
    epsi_1 = h_1 - np.cos(q)
    Eq = np.sqrt((h_1 - 1) ** 2 + 2 * h_1 * (1 - np.cos(q)))
    theta_q0 = np.zeros(Nq)
    theta_q1 = np.zeros(Nq)

    for i in range(Nq):
        if epsi_0[i] > 0:
            theta_q0[i] = math.atan(sq[i] / epsi_0[i])
        else:
            theta_q0[i] = np.pi - math.atan(abs(sq[i] / epsi_0[i]))
        if epsi_1[i] > 0:
            theta_q1[i] = math.atan(sq[i] / epsi_1[i])
        else:
            theta_q1[i] = np.pi - math.atan(abs(sq[i] / epsi_1[i]))

    alpha_q = 1 / 2 * (theta_q0 - theta_q1)
    Sz = np.zeros(Nt)
    for j in range(Nt):
        sz = 0
        for i in range(Nq):
            a = np.cos(2 * alpha_q[i]) * np.cos(theta_q1[i])
            b = np.sin(2 * alpha_q[i]) * np.sin(theta_q1[i]) * np.cos(4* t[j] * Eq[i])
            sz = sz + a - b
        Sz[j] = sz / Nq

    plt.subplot(2, 1, 1)
    plt.plot(t, Sz,color='lightcoral', linewidth=1.5, linestyle='-', label='Exact_h:0.9,->0.5')
    plt.plot(time,sigma_zs,'+',color='mediumturquoise',markersize=1.5,label='DMRG-infinite_h:0.9->0.5')
    plt.legend(loc='best')
    plt.xlabel('h/J', fontsize=13)
    plt.ylabel('sigma_z', fontsize=13)
    plt.subplot(2, 1, 2)
    plt.xlabel('t', fontsize=13)
    plt.ylabel('entanglement', fontsize=13)
    plt.plot(time, S, linewidth=1.5)
    plt.show()




