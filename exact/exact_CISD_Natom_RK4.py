import numpy as np
import time
import scipy.sparse
import scipy.sparse.linalg
from numba import jit, prange

with open('input_threelevel.txt') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        exec(str(line), globals())

start_time_all = time.time()

#Hamiltonian
start_time_H=time.time()

#assign index
N=int(N)
w = np.append(0, w)
alpha = np.append(0, alpha)

#assign index for modes
NumDiff2modes=int((2*N)*(2*N-1)/2)
num_modesbasis=int(4*N+1+NumDiff2modes)
indexab=np.zeros((num_modesbasis, 2))
iter= int(1+4*N)
indexab[0]=((-1,-1))
for i in range(int(2 * N)):
    indexab[i+1] = ((i, -1))
    indexab[int(i+2*N+1)] = ((i, i))
    for j in range(i + 1, int(2 * N)):
        indexab[iter] = ((i,j))
        iter = iter+1

#assign index with atoms
total_number=int(energy.size**num_atom*num_modesbasis)
index = np.zeros((total_number, int(num_atom+2)))

for n in range(num_atom):
    iii = 0
    for i in range(energy.size**(num_atom-n)):
        index[int(iii):int(iii+(n+1)*(num_modesbasis)), int(num_atom-1-n)] = i%energy.size
        if n == 0:
            index[int(iii):int(iii + num_modesbasis), int(num_atom):int(num_atom+2)] =indexab
        iii += (n+1)*(num_modesbasis)

@jit(nopython=True, fastmath=True)#, parallel=True)
def get_H_nomatrix():
    Hindex=np.array([[-1,-1]], dtype='int32')
    H=np.array([1], dtype='float64')
    #H=np.array(-1, dtype='float') #just to initiate H
    for i in range(int(2*total_number/3)):
        Hii = 0.0
        field_i=int(index[i][num_atom] + 1)
        field_ii = int(index[i][num_atom+1] + 1)
        Hii = hbar * w[field_i] + hbar * w[field_ii]
        atom_i = int(index[i][0])
        Hii += energy[atom_i]
        # for ni in range(num_atom):
        #     atom_i = int(index[i][ni])
        #     Hii += energy[atom_i]
            # if self_pol == 1:
            #     if atom_i == 0:
            #         Hii += + 0.5 * (nu[0]**2)*((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
            #         np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
            #         np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
            #     elif atom_i == 1:
            #         Hii += 0.5 * (nu[0] ** 2 + nu[1] ** 2) * ((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
            #         np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
            #         np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
            #     else:
            #         Hii += + 0.5 * (nu[1] ** 2) * ((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
            #         np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
            #         np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
        nowHindex=np.array([[i,i]], dtype='int32')
        Hindex = np.vstack((Hindex, nowHindex))
        H = np.append(H, Hii)

        for j in range(i+1, total_number):
            Hij=0.0
            field_j = int(index[j][num_atom] + 1)
            field_jj = int(index[j][num_atom+1] + 1)
            nj=0
            atom_j = int(index[j][nj])
            # for nj in range(num_atom):
            #     atom_i = int(index[i][nj])
            #     isum = np.sum(index[i][:-2])
            #     atom_j = int(index[j][nj])
            #     jsum = np.sum(index[j][:-2])
                # if self_pol == 1:
                #     if atom_i - atom_j == -2 and field_i == field_j and field_ii == field_jj:
                #             Hij += 0.5 * nu[atom_i]*nu[atom_i+1]*((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
                #     np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
                #     np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
                #     if atom_i - atom_j == 2 and field_i == field_j and field_ii == field_jj:
                #             Hij += + 0.5 * nu[atom_j]*nu[atom_j+1]*((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
                #     np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
                #     np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
                #if np.abs(isum-jsum) == 1:
            if atom_i - atom_j == -1:
                if RWA == 0:
                    # For cedagger_cg_adagger on atom 1
                    if field_i == 0 and field_j != 0 and field_ii == field_jj:
                        Hij += nu[atom_i] * np.sqrt((w[field_j]) / (epsilon * l)) * np.sin(
                            np.pi * alpha[field_j] * r_atom[nj] / l)
                    if field_ii == 0 and field_jj != 0 and field_i == field_j:
                        if field_j == field_jj:
                            Hij += np.sqrt(2) * nu[atom_i] * np.sqrt((w[field_jj]) / (epsilon * l)) * np.sin(
                                np.pi * alpha[field_jj] * r_atom[nj] / l)
                        else:
                            Hij += nu[atom_i] * np.sqrt((w[field_jj]) / (epsilon * l)) * np.sin(
                                np.pi * alpha[field_jj] * r_atom[nj] / l)
                    if field_ii == 0 and field_j != 0 and field_i == field_jj and field_i != field_ii and field_j != field_jj:
                        Hij += nu[atom_i] * np.sqrt((w[field_j]) / (epsilon * l)) * np.sin(
                            np.pi * alpha[field_j] * r_atom[nj] / l)
                # For cedagger_cg_a on atom 1
                if field_i != 0 and field_j == 0 and field_ii == field_jj:
                    Hij += nu[atom_i] * np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
                        np.pi * alpha[field_i] * r_atom[nj] / l)
                if field_ii != 0 and field_jj == 0 and field_i == field_j:
                    if field_i == field_ii:
                        Hij += np.sqrt(2)*nu[atom_i] * np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
                            np.pi * alpha[field_ii] * r_atom[nj] / l)
                    else:
                        Hij += nu[atom_i] * np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
                            np.pi * alpha[field_ii] * r_atom[nj] / l)
                if field_i != 0 and field_jj == 0 and field_ii == field_j and field_j != field_jj and field_i != field_ii:
                    Hij += nu[atom_i] * np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
                        np.pi * alpha[field_i] * r_atom[nj] / l)
                if Hij != 0.0:
                    nowHindex = np.array([[i, j]], dtype='int32')
                    Hindex = np.vstack((Hindex, nowHindex))
                    H = np.append(H, Hij)
    for i in range(int(2 * total_number / 3), total_number):
        Hii = 0.0
        field_i = int(index[i][num_atom] + 1)
        field_ii = int(index[i][num_atom + 1] + 1)
        Hii = hbar * w[field_i] + hbar * w[field_ii]
        atom_i = int(index[i][0])
        Hii += energy[atom_i]
        # for ni in range(num_atom):
        #     atom_i = int(index[i][ni])
        #     Hii += energy[atom_i]
        # if self_pol == 1:
        #     if atom_i == 0:
        #         Hii += + 0.5 * (nu[0]**2)*((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
        #         np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
        #         np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
        #     elif atom_i == 1:
        #         Hii += 0.5 * (nu[0] ** 2 + nu[1] ** 2) * ((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
        #         np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
        #         np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
        #     else:
        #         Hii += + 0.5 * (nu[1] ** 2) * ((np.sqrt((w[field_i]) / (epsilon * l)) * np.sin(
        #         np.pi * alpha[field_i] * r_atom[ni] / l)) ** 2+(np.sqrt((w[field_ii]) / (epsilon * l)) * np.sin(
        #         np.pi * alpha[field_ii] * r_atom[ni] / l)) ** 2)
        nowHindex = np.array([[i, i]], dtype='int32')
        Hindex = np.vstack((Hindex, nowHindex))
        H = np.append(H, Hii)
    return Hindex, H

Hindex, H = get_H_nomatrix()
Hindex = np.delete(Hindex, 0, 0)
H = np.delete(H, 0)

all_ini_wavefn=np.zeros(total_number, dtype='complex128')
basis_states=all_ini_wavefn.size

H = scipy.sparse.coo_matrix((np.concatenate((H,H)),
                             (np.concatenate((Hindex[:,0],Hindex[:,1])),
                              np.concatenate((Hindex[:,1],Hindex[:,0])))))
H = H.tocsr()

end_time_H=time.time()
print('Hamiltonian generated. Shape='+ str(H.shape) + ', 2N=' + str(2*N))
print('Running Wall Time = %10.3f second' % (end_time_H - start_time_H))

# @jit(nopython=True, fastmath=True)
def get_eig(H):
    sparse_H=scipy.sparse.csc_matrix(H)
    evals, evecs = scipy.sparse.linalg.eigsh(sparse_H, k=1, which='SA')
    return evals, evecs

#@jit(nopython=True, fastmath=True)
def RK4(H, wavefn, dt):
    K1 = (-1j/hbar * H @ wavefn)
    K2 = (-1j/hbar * H @ (wavefn + 0.5 * dt * K1))
    K3 = (-1j / hbar * H @ (wavefn + 0.5 * dt * K2))
    K4 = (-1j / hbar * H @ (wavefn + dt * K3))
    wavefn = wavefn + dt * 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    return wavefn

start_time_eigh=time.time()
evals, evecs = get_eig(H)
end_time_eigh=time.time()

print('Found the lowest eigenstate. Running Wall Time = %10.3f second' % (end_time_eigh - start_time_eigh))

start_time_run=time.time()
#all_ini_wavefn[np.where((index[:,0] == 1) & (index[:,1] == -1))]=1 # |e, 0>
#all_ini_wavefn[np.where((index[:,0] == 1) & (index[:,1] == 0) & (index[:,2] == -1))]=1 # |e, g, 0>
#all_ini_wavefn=evecs[:,0] # "Real ground state"

all_ini_wavefn[np.where((index[:,0] == 1))]\
    = evecs[:,0][np.where((index[:,0] == 0))]
all_ini_wavefn[np.where((index[:,0] == 2))]\
    = evecs[:,0][np.where((index[:,0] == 1))]
all_ini_wavefn=all_ini_wavefn/np.linalg.norm(all_ini_wavefn)

wavefn=all_ini_wavefn
wavefn_save=wavefn
for t in steps:
    wavefn = RK4(H, wavefn, dt)
    if t % savestep == 0:
        wavefn_save = np.vstack([wavefn_save, wavefn])
    if t == steps[-1]:
        wavefnend = wavefn

end_time_all = time.time()

np.save('index.npy', index)
np.savez_compressed('rho.npz', rho=wavefn_save)
#np.savez_compressed('rho_eig.npz', rho=wavefn_eigs_save)
#np.savetxt('rho.csv', wavefn_save, delimiter=',')
np.savetxt('final_rho.csv', wavefnend, delimiter=',')
print('Finished time evolution. Running Wall Time = %10.3f second' % (end_time_all - start_time_run))
print('Calculation Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))
