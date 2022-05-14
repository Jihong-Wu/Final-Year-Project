import numpy as np
from scipy.linalg import svd
import scipy.sparse
import scipy.sparse.linalg.eigen.arpack as arp
import matplotlib.pyplot as plt


class SimpleHeff(scipy.sparse.linalg.LinearOperator):
    def __init__(self, LP, RP, W1, W2):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W1 = W1  # wL wC i i*
        self.W2 = W2  # wC wR j j*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d1, d2 = W1.shape[2], W2.shape[2]
        self.theta_shape = (chi1, d1, d2, chi2)  # vL i j vR
        self.shape = (chi1 * d1 * d2 * chi2, chi1 * d1 * d2 * chi2)
        self.dtype = W1.dtype

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>.

        This function is used by :func:scipy.sparse.linalg.eigen.arpack.eigsh` to diagonalize
        the effective Hamiltonian with a Lanczos method, withouth generating the full matrix."""
        x = np.reshape(theta, self.theta_shape)  # vL i j vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i j vR
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))  # vL [wL*] [i] j vR, [wL] wC i [i*]
        x = np.tensordot(x, self.W2, axes=([3, 1], [0, 3]))  # vL [j] vR [wC] i, [wC] wR j [j*]
        x = np.tensordot(x, self.RP, axes=([1, 3], [0, 1]))  # vL [vR] i [wR] j, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x


def update_RP(i,RPs,Bs,mpo):
    j= i-1
    RP=RPs[i]
    B=Bs[i]
    Bc=B.conj()
    W=mpo[i]
    RP=np.tensordot(B,RP,axes=[2,0])
    RP=np.tensordot(RP,W,axes=[[1,2],[3,1]])
    RP=np.tensordot(RP,Bc,axes=[[1,3],[2,1]])
    RPs[j]=RP
    return RPs


def update_LP(i,LPs,Bs,Ss,mpo):
    j=i+1
    LP=LPs[i]
    B=Bs[i]
    G=np.tensordot(np.diag(Ss[i]),B,axes=[1,0])
    A=np.tensordot(G,np.diag(Ss[j]**(-1)),axes=[2,0])
    Ac=A.conj()
    W=mpo[i]
    LP=np.tensordot(LP,A,axes=[2,0])
    LP=np.tensordot(W,LP,axes=[[0,3],[1,2]])
    LP=np.tensordot(Ac,LP,axes=[[0,1],[2,1]])
    LPs[j]=LP
    return LPs


def truncate_theta(theta,chi,eps):
    chivL,dL,dR,chivR=theta.shape
    theta=np.reshape(theta,[chivL*dL,chivR*dR])
    X,Y,Z=svd(theta,full_matrices=False)
    chivC=min(chi,np.sum(Y>eps))
    piv=np.argsort(Y)[::-1][:chivC]
    X,Y,Z=X[:,piv],Y[piv],Z[piv,:]

    S=Y/np.linalg.norm(Y)
    A=np.reshape(X,[chivL,dL,chivC])
    B=np.reshape(Z,[chivC,dR,chivR])
    return A,S,B


def update_bond(i,Bs,Ss,LPs,RPs,mpo,chi):
    j=i+1
    Heff=SimpleHeff(LPs[i],RPs[j],mpo[i],mpo[j])
    x1=np.tensordot(np.diag(Ss[i]),Bs[i],[1,0])
    x2=np.tensordot(x1,Bs[j],[2,0])
    theta0=np.reshape(x2,[Heff.shape[0]])
    e,v=arp.eigsh(Heff,k=1,which='SA',return_eigenvectors=True,v0=theta0)
    theta=np.reshape(v[:,0],Heff.theta_shape)

    eps=1.e-10
    Ai,Sj,Bj=truncate_theta(theta,chi,eps)
    Gi=np.tensordot(np.diag(Ss[i]**(-1)),Ai,axes=[1,0])
    Bs[i]=np.tensordot(Gi,np.diag(Sj),axes=[2,0])
    Ss[j]=Sj
    Bs[j]=Bj
    LPs=update_LP(i,LPs,Bs,Ss,mpo)
    RPs=update_RP(j,RPs,Bs,mpo)


    return Bs,Ss,LPs,RPs


def Sz_value(L,Bs,Ss):
    result=[]
    sz=np.array([[1., 0.], [0., -1.]])
    c=L//2
    theta_i = np.tensordot(np.diag(Ss[c]), Bs[c], axes=[1, 0])
    a = np.tensordot(sz, theta_i, axes=[1, 1])
    result.append(np.tensordot(theta_i.conj(), a, axes=[[0, 1, 2], [1, 0, 2]]))
    return np.real_if_close(result)
'''
    for i in range(L):
        theta_i=np.tensordot(np.diag(Ss[i]),Bs[i],axes=[1,0])
        a=np.tensordot(sz,theta_i,axes=[1,1])
        result.append(np.tensordot(theta_i.conj(),a,axes=[[0,1,2],[1,0,2]]))

    return np.real_if_close(result)
'''


def solve_E0(L,Bs,Ss,J,h):
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id=np.eye(2)
    H_bonds=[]
    for i in range(L-1):
        hL=hR=0.5*h
        if i==0:
            hL=h
        if i+1==L-1:
            hR=h
        H_bond=-J*np.kron(sx,sx)-hL*np.kron(sz,id)-hR*np.kron(id,sz)
        H_bonds.append(np.reshape(H_bond,[2,2,2,2]))
    result=[]
    for i in range(L-1):
        x1=np.tensordot(np.diag(Ss[i]),Bs[i],[1,0])
        j=i+1
        theta=np.tensordot(x1,Bs[j],[2,0])
        op_theta=np.tensordot(H_bonds[i],theta,axes=[[2,3],[1,2]])
        result.append(np.tensordot(theta.conj(),op_theta,[[0,1,2,3],[2,0,1,3]]))
    return np.real_if_close(result)


def entropy(Ss,j):
    result = []
    S=Ss[j].copy()
    S[S<1.e-20]=0
    S2=S*S
    result.append(-np.sum(S2*np.log(S2)))
    return result


def DMRG(L,J,h,Bs,Ss,mpo,chi,sweeps):
    LPs=[None]*L
    RPs=[None]*L
    LP=np.zeros([1,3,1],dtype=float)
    RP=np.zeros([1,3,1],dtype=float)
    LP[:,0,:]=np.eye(1)
    RP[:,2,:]=np.eye(1)
    LPs[0]=LP
    RPs[-1]=RP

    #initialize RPs
    for i in range(L-1,1,-1):
        RPs=update_RP(i,RPs,Bs,mpo)

    #do sweeps
    for j in range(sweeps):
        for i in range(L - 2):
            Bs, Ss, LPs, RPs = update_bond(i, Bs, Ss, LPs, RPs, mpo, chi)

        for i in range(L - 2, 0, -1):
            Bs, Ss, LPs, RPs = update_bond(i, Bs, Ss, LPs, RPs, mpo, chi)

    return Bs,Ss



if __name__ == "__main__":
    #TFIM
    L=20; J=1.; h=0.5;
    chi=30; sweeps=30;
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.eye(2)

    #initialize MPS
    B= np.zeros([1, 2, 1], dtype=float)
    B[0, 0, 0] = 1.
    S=np.ones([1], dtype=float)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]

    #initialize MPO
    mpo = []
    for i in range(L):
        w=np.zeros((3,3,2,2),dtype=float)
        w[0,0]=w[2,2]=id
        w[0,1]=sx
        w[0,2]=-h*sz
        w[1,2]=-J*sx
        mpo.append(w)

    Bs,Ss=DMRG(L,J,h,Bs,Ss,mpo,chi,sweeps)
    #calculations
    print("DMRG_finite","L=",L," h=",h)
    sigma_z=np.sum(Sz_value(L,Bs,Ss))
    print("<sigma_z>=",sigma_z)
    E0=np.sum(solve_E0(L,Bs,Ss,J,h))
    print("E0=",E0/L)




