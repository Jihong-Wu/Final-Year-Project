import numpy as np
from dmrg_finite import DMRG, solve_E0, Sz_value, update_RP
from scipy.linalg import expm
from scipy.linalg import qr
import matplotlib.pyplot as plt


def newSz_value(L,Cs):
    result=[]
    sz=np.array([[1., 0.], [0., -1.]])
    c=L//2
    #theta_i = np.tensordot(Bs[c],Cs[c], axes=[2, 0])
    theta_i = Cs[c]
    a = np.tensordot(sz, theta_i, axes=[1, 1])
    result.append(np.tensordot(theta_i.conj(), a, axes=[[0, 1, 2], [1, 0, 2]]))
    return np.real_if_close(result)


#def newupdate_RP(i,RPs,Bli,mpo):


def newupdate_LP(i,LPs,Bli,mpo):
    j=i+1
    LP=LPs[i]
    Ac=Bli.conj()
    W=mpo[i]
    LP=np.tensordot(LP,Bli,axes=[2,0])
    LP=np.tensordot(W,LP,axes=[[0,3],[1,2]])
    LP=np.tensordot(Ac,LP,axes=[[0,1],[2,1]])
    LPs[j]=LP
    return LPs


def doKeff(Heff,A,dt):
    chi1,d,chi2=Heff.shape[0],Heff.shape[1],Heff.shape[2]
    Acon=A.conj()
    x1=np.tensordot(Heff,A,axes=[[0,1],[0,1]])
    x2=np.tensordot(x1,Acon,axes=[[1,2],[0,1]])
    x3=np.transpose(x2,[0,2,1,3])
    U=np.reshape(x3,[chi2*chi2,chi2*chi2])
    K=expm(1j*dt*U)
    Keff=np.reshape(K,[chi2,chi2,chi2,chi2])
    return Keff


def doHeff(LP,RP,W,dt):
    chi1,chi2=LP.shape[0], RP.shape[0]
    d=W.shape[2]
    x1=np.tensordot(LP,W,axes=[1,0])
    x2=np.tensordot(x1,RP,axes=[2,1])
    x3=np.transpose(x2,[0,2,4,1,3,5])
    U=np.reshape(x3,[chi1*d*chi2,chi1*d*chi2])
    H=expm(-1j*dt*U)
    Heff=np.reshape(H,[chi1,d,chi2,chi1,d,chi2])
    return Heff


def update_bond1(i, Bs, Ss, Cs, LPs, RPs, mpo, dt):
    #one-site
    #if i!=(L-1):
    #Heff = doHeff(LPs[i], RPs[i], mpo[i], dt/2 )
    #if i==(L-1):
    Heff = doHeff(LPs[i], RPs[i], mpo[i], dt)

    j = i + 1
    Bci = np.tensordot(Ss[i], Bs[i], axes=[1, 0])
    # evolve forward in time
    Bci_t = np.tensordot(Bci, Heff, axes=[[0, 1, 2], [0, 1, 2]])
    Cs[i] = Bci_t
    if i!=(L-1):
        #QR orthogonal decomposition
        chi1, d, chi2 = Bci_t.shape[0], Bci_t.shape[1], Bci_t.shape[2]
        x = np.reshape(Bci_t, [chi1 * d, chi2])
        Bli_t, Sj_t = qr(x)
        piv=range(chi2)
        Bli_t=Bli_t[:,piv] #chi1*d*chi2
        Sj_t=Sj_t[piv,:] #chi2*chi2
        Bli_t=np.reshape(Bli_t,[chi1,d,chi2])
        #evolve backwards in time
        Keff=doKeff(Heff,Bli_t,dt/2)
        Sj=np.tensordot(Sj_t,Keff,axes=[[0,1],[0,1]])
        Ss[j] = Sj
        Bs[i]=Bli_t
        LPs = newupdate_LP(i, LPs, Bli_t, mpo)


    if i==(L-1):
        Bs[i] = Bci_t
        RPs = update_RP(i, RPs, Bs, mpo)
    return Bs,Ss,Cs,LPs,RPs


def update_bond2(i, Bs, Ss, LPs, RPs, mpo, dt):
    j=i+1
    chi1,d,chi2=Bs[j].shape[0],Bs[j].shape[1],Bs[j].shape[2]
    x=np.reshape(Bs[j],[chi1,d*chi2])
    Bri_t,Sj_t= qr(x)
    piv=range(chi2)
    Bri_t=Bri_t[:,piv]
    Sj_t=Sj_t[piv,:]
    Sj_t=np.reshape(Sj_t,[chi2,chi2])
    #Evolve backwards in time
    Heff=doHeff(LPs[i],RPs[i],mpo[i],dt/2)
    Keff=doKeff(Heff,Bs[i],dt/2)
    print(Bs[j].shape)
    '''
    Sj=np.tensordot(Sj_t,Keff,axes=[[0,1],[0,1]])
    Ss[j]=Sj
    Bci=np.tensordot(Bs[i],Sj,axes=[1,0])
    Bci_t=np.tensordot(Bci,Heff,axes=[[0,1,2],[0,1,2]])
    Bs[i]=Bci_t
    '''


def TDVP(L,Bs,Ss,mpo,dt):
    LPs = [None] * L
    RPs = [None] * L
    D = mpo[0].shape[0]
    d = Bs[0].shape[0]
    LP = np.zeros([d, D, d], dtype=float)
    RP = np.zeros([d, D, d], dtype=float)
    LP[:, 0, :] = np.eye(d)
    RP[:, D - 1, :] = np.eye(d)
    LPs[0] = LP
    RPs[-1] = RP



    # initialize RPs
    for i in range(L - 1, 0, -1):
        RPs = update_RP(i, RPs, Bs, mpo)

    Cs=[None]*(L)
    #Bs, Ss, LPs, RPs = update_bond(0, Bs, Ss, LPs, RPs, mpo, dt)
    for i in range(L):
        Bs,Ss,Cs,LPs,RPs=update_bond1(i, Bs, Ss, Cs, LPs, RPs, mpo, dt)

    # update_bond2(L-2, Bs, Ss, LPs, RPs, mpo, dt)
    #for i in range(L-2,-1,-1):
        #update_bond2(i, Bs, Ss, LPs, RPs, mpo, dt)



    sigma_z = np.sum(newSz_value(L, Cs))
    print("t=",dt,"<sigma_z>=",sigma_z)


'''
    #evolution  一个sweep时间增加怎么算？
    for i in range(L-2):
        Bs, Ss, LPs, RPs = update_bond(i, Bs, Ss, LPs, RPs, mpo, dt)

    for i in range(L - 2, 0, -1):
        Bs, Ss, LPs, RPs = update_bond(i, Bs, Ss, LPs, RPs, mpo, dt)
'''

if __name__ == "__main__":
    L = 20; J = 1.; h1 = 4; h2 = 2
    chi = 30; sweeps = 30
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
        w[0, 2] = -h1 * sz
        w[1, 2] = -J * sx
        mpo.append(w)

    # find ground state
    Bs, Ss = DMRG(L, Bs, Ss, mpo, chi, sweeps)
    sigma_zs = []; energy = [];
    print("TDVP_finite " "L=", L, " h=", h1)
    sigma_z = np.sum(Sz_value(L, Bs, Ss))
    E0 = np.sum(solve_E0(L, Bs, Ss, J, h1))
    sigma_zs.append(sigma_z)
    energy.append(E0)
    print("t={t:f}, E0={E0:.5f},<sigma_z>={sigmaz:.5f}".format(t=0, E0=E0 / L, sigmaz=sigma_z))

    #tdvp real time evolution
    # new MPO
    mpo = []
    for i in range(L):
        w = np.zeros((3, 3, 2, 2), dtype=float)
        w[0, 0] = w[2, 2] = id
        w[0, 1] = sx
        w[0, 2] = -h2 * sz
        w[1, 2] = -J * sx
        mpo.append(w)

    tmax=0.01;
    dt=0.01;
    Nsteps=int(tmax/dt)

    for i in range(L):
        Ss[i]=np.diag(Ss[i])

    for i in range(Nsteps):
        TDVP(L,Bs,Ss,mpo,dt)
    #print(Bs[L-1].shape)









