import numpy as np
import time
from numba import jit, prange

with open('input_threelevel.txt') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        exec(str(line), globals())
start_time_all = time.time()
name='rho'

NumDiff2modes=int((2*N)*(2*N-1)/2)
wavefn_save=np.load(name+'.npz')['rho']
index=np.load('index.npy')

r = np.linspace(0,l,r_resolution)

# generate index for all basis
basis_index=np.zeros((int(8*N+2+2*NumDiff2modes),3))

def get_mode_func(alpha):
    mode_func = np.sqrt((w[:,np.newaxis]) / (epsilon*l) ) \
                          * np.sin(np.pi*alpha[:,np.newaxis]*r/l)
    return mode_func

# generate index for alpha^1beta^1
index_2 = [0, 0]
for i in range(int(2 * N)):
    for j in range(i + 1, int(2 * N)):
        index_2 = np.vstack([index_2, [i, j]])

# Calculation part
mode_func = get_mode_func(alpha)  # has a shape of ( # of modes, r_res)

@jit(nopython=True, fastmath=True)#, parallel=True)
def calc_ele_int(wavefn_save):
    wavefn_conj = np.conjugate(wavefn_save)
    ele_intensity = np.zeros((r.size, int(tmax / intens_save_t) + 1))
    index_t = 0
    #for t in range(0, int(tmax + 1), intens_save_t):
    for t in range(int(tmax / intens_save_t) + 1):
        g0 = np.zeros(r.size) + 0j
        g1 = np.zeros(r.size) + 0j
        g2 = np.zeros(r.size) + 0j
        g11= np.zeros(r.size) + 0j
        t_index = int(t*intens_save_t / dt / savestep)
        # Calculate |g,0>
        # to |g, 0>
        # g0=np.sum(wavefn_conj[t_index, 0]* mode_func**2 *wavefn_save[t_index, 0], axis=0)
        # to |g, alpha2>
        for i in range(int(2 * N + 1), int(2 * 2 * N) + 1):
            index = int(i - (2 * N + 1))
            g0 += np.sqrt(2)*wavefn_conj[t_index, i]*mode_func[index]**2*wavefn_save[t_index, 0]
        # to |g, alpha1beta1>
        for i in range(2*int(2*N)+1,2*int(2*N)+NumDiff2modes+1):
            index = i - (4*N+1)
            index_for_w = index_2[int(index+1)]
            g0+=2*wavefn_conj[t_index, i]*mode_func[index_for_w[1]]*mode_func[index_for_w[0]]*wavefn_save[t_index, 0]

        # Calculate |g,alpha1>
        for i in range(1, int(2 * N + 1)):
            index = i - 1
            # to |g, alpha1>
            #g1 = g1 + 2 * wavefn_conj[t_index, i] * mode_func[index]**2 * wavefn_save[t_index, i]
            # to |g, beta1>
            for j in range(1, int(2 * N + 1)):
                index_j = j - 1
                g1+= 2*wavefn_conj[t_index, i] * mode_func[index] * \
                        mode_func[index_j] * wavefn_save[t_index, j]

        # Calculate |g,alpha2>
        for i in range(int(2 * N + 1), int(2 * 2 * N) + 1):
            index = int(i - (2 * N + 1))
            # to |g, 0>
            g2 += np.sqrt(2) * wavefn_conj[t_index, 0] * mode_func[index] ** 2 * wavefn_save[t_index, i]
            # to |g, alpha2>
            g2 += 4 * wavefn_conj[t_index, i] * mode_func[index] ** 2 * wavefn_save[t_index, i]
            # to |g, alpha1beta1>
            for j in range(2 * int(2 * N) + 1, 2 * int(2 * N) + NumDiff2modes + 1):
                index_j = j - (4 * N + 1)
                index_for_w = index_2[int(index_j + 1)]
                if alpha[index] == alpha[index_for_w[0]]:
                    g2 = g2 + 2*np.sqrt(2) * wavefn_conj[t_index, j] * mode_func[index]\
                         * mode_func[index_for_w[1]]* wavefn_save[t_index, i]
                if alpha[index] == alpha[index_for_w[1]]:
                    g2 = g2 + 2*np.sqrt(2) * wavefn_conj[t_index, j] * mode_func[index]\
                         * mode_func[index_for_w[0]]* wavefn_save[t_index, i]

        # Calculate |g, alpha1beta1>
        for i in range(2 * int(2 * N) + 1, 2 * int(2 * N) + NumDiff2modes + 1):
            index = i - (4 * N + 1)
            index_for_w = index_2[int(index + 1)]
            # to |g,0>
            g11 += 2* wavefn_conj[t_index, i] * mode_func[index_for_w[0]]\
                         * mode_func[index_for_w[1]]* wavefn_save[t_index, 0]
            # to |g, alpha2>
            for j in range(int(2 * N + 1), int(2 * 2 * N) + 1):
                index_j = int(j - (2 * N + 1))
                if index_j == alpha[index_for_w[0]]:
                    g11 += 2*np.sqrt(2) * wavefn_conj[t_index, j] * mode_func[index_for_w[1]] \
                     * mode_func[index_j] * wavefn_save[t_index, i]
                if index_j == alpha[index_for_w[1]]:
                    g11 += 2*np.sqrt(2) * wavefn_conj[t_index, j] * mode_func[index_for_w[0]] \
                     * mode_func[index_j] * wavefn_save[t_index, i]
            # to |g, alpha1beta1>
            #g11 = g11 + 2 * wavefn_conj[t_index, i] * mode_func[index_for_w[0]]**2 * wavefn_save[t_index, i]
            #g11 = g11 + 2 * wavefn_conj[t_index, i] * mode_func[index_for_w[1]]** 2 * wavefn_save[t_index, i]
            # to |g, alpha1 i1>
            for j in range(2 * int(2 * N) + 1, 2 * int(2 * N) + NumDiff2modes + 1):
                index_j = j - (4 * N + 1)
                index_for_w_j = index_2[int(index_j + 1)]
                if alpha[index_for_w[0]] == alpha[index_for_w_j[0]]:
                    g11 += 2*wavefn_conj[t_index, j] * mode_func[index_for_w[1]] \
                         * mode_func[index_for_w_j[1]] * wavefn_save[t_index, i]
                if alpha[index_for_w[0]] == alpha[index_for_w_j[1]]:
                    g11 += 2*wavefn_conj[t_index, j] * mode_func[index_for_w[1]] \
                         * mode_func[index_for_w_j[0]] * wavefn_save[t_index, i]

                if alpha[index_for_w[1]] == alpha[index_for_w_j[0]]:
                    g11 += 2*wavefn_conj[t_index, j] * mode_func[index_for_w[0]] \
                         * mode_func[index_for_w_j[1]] * wavefn_save[t_index, i]
                if alpha[index_for_w[1]] == alpha[index_for_w_j[1]]:
                    g11 += 2*wavefn_conj[t_index, j] * mode_func[index_for_w[0]] \
                         * mode_func[index_for_w_j[0]] * wavefn_save[t_index, i]

        ele_intensity[:, index_t] = np.real(g0+g1+g2+g11)
        index_t += 1
    return ele_intensity

start_time_calc=time.time()

ele_intensity = calc_ele_int(wavefn_save[:,:int(wavefn_save.shape[1]/2)]) \
                    + calc_ele_int(wavefn_save[:,int(wavefn_save.shape[1]/2):])

end_time_calc=time.time()
print('Finished calculating field intensity')
print('Running Wall Time = %10.3f second' % (end_time_calc - start_time_calc))

np.savetxt('ele_intensity_' + name + '.csv', ele_intensity, delimiter=',')

end_time_all=time.time()
print('Calculation Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))
print('current file name:' + str(name))

#@jit(nopython=True, fastmath=True)#, parallel=True)
def calc_pho_num(wavefn_save):
    rho=np.abs(wavefn_save)**2
    photon_number_each=np.zeros((int(2*N),int(tmax / intens_save_t) + 1))
    index_t = 0
    for t in range(int((tmax + 1) / intens_save_t)):
        t_index = int(t * intens_save_t / dt / savestep)
        for i in range(int(2*N)):
            photon_number_each[i, index_t] = np.sum(rho[t_index, np.where((index[:,1] == i))[0]])\
                                                    + np.sum(rho[t_index, np.where((index[:,2] == i))[0]])
        index_t = index_t + 1
    return photon_number_each

start_time_pho=time.time()

photon_number_each=calc_pho_num(wavefn_save)
np.savetxt('photon_number_each.csv', photon_number_each, delimiter=',')

end_time_pho=time.time()
print('Calculation of photon number Finished. Running Wall Time = %10.3f second' % (end_time_pho - start_time_pho))