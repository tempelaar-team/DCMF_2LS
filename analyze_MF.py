import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import time

# input file reading part
with open('input.txt') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        exec(str(line), globals())

start_time_all = time.time()
au_to_micron=5.2917724900001E-5

#loading part
start_time_loading = time.time()
square_wavefn=np.loadtxt(calcdir + '/rho.csv', delimiter=',')
all_QaQb=np.load(calcdir + '/q1q2.npz')['Q1Q2']
all_P_square=np.loadtxt(calcdir + '/p2.csv', delimiter=',', ndmin=2)
all_E=np.loadtxt(calcdir + '/E.csv', delimiter=',')
end_time_loading = time.time()

print('Finished Loading Files. Running Wall Time = %10.3f second' % (end_time_loading - start_time_loading))

tmax = (square_wavefn[:,0].size - 1) * savestep * dt
num_trj=int(square_wavefn[0,0]+square_wavefn[0,1])
avg_E=all_E/num_trj

#calculating average of atomic population
avg_wavefn=square_wavefn/num_trj

#calculating proton number
all_QaQb_avg=all_QaQb/num_trj
all_Psquare_avg=all_P_square/num_trj
photon_number=np.zeros(int(tmax/dt/savestep+1))

start_time_ph=time.time()
for t in np.arange(int(tmax/dt/savestep+1)):
    photon_number[t]=1/2 * np.sum(all_Psquare_avg[:,t]/(hbar*w) + (w*np.diagonal(all_QaQb_avg)[t,:])/(hbar) - 1)
end_time_ph=time.time()
print('Finished Calculating Photon Number. Running Wall Time = %10.3f second' % (end_time_ph - start_time_ph))

# calculate electric-field intensity
# at given time and given distance
start_time_ele=time.time()
r = np.linspace(0,l,r_resolution)

# for a given time (t= 100 = step of tmax/dt)
t_all_QaQb_avg=np.zeros([int(2*N),int(2*N),int(tmax/intens_save_t)+1])
photon_number_each=np.zeros([int(2*N), int(tmax/intens_save_t)+1])

index_t=0
for t in np.arange(0, tmax + 1, intens_save_t):
    t_all_QaQb_avg[:, :, int(index_t)] = all_QaQb_avg[:, :, int(t / dt / savestep)]
    photon_number_each[:, int(index_t)] = 1/2*(all_Psquare_avg[:,int(t / dt / savestep)]/(hbar*w)
                                               + (w*np.diagonal(all_QaQb_avg)[int(t / dt / savestep),:])/(hbar))
    index_t=index_t+1

wawb=np.outer(w,w)
xi=np.zeros((int(2*N),r.size))
xia_xib=np.zeros((int(2*N),int(2*N),r.size))
ele_intensity=np.zeros((r.size,int(tmax/intens_save_t)+1))

# For each r
print('Calculating field intensity...')
xi = np.sqrt(2/l)*np.sin(alpha[:,np.newaxis]*np.pi/l * r)
for i in np.arange(0,r.size,1):
    xia_xib[:, :, i] = np.outer(xi[:, i], xi[:, i])

index_t = 0
for t in np.arange(0, tmax + 1, intens_save_t):
    ele_intensity[:, index_t] = 1 / epsilon * np.sum(wawb[:,:,np.newaxis] *
                                                     xia_xib * t_all_QaQb_avg[:, :, index_t][:,:,np.newaxis], axis=((0,1)))\
                - np.sum(xi**2*hbar*w[:,np.newaxis]/(2*epsilon), axis=0)

    index_t = index_t+1
end_time_ele=time.time()
print('Finished Calculating field intensity. Running Wall Time = %10.3f second' % (end_time_ele - start_time_ele))

#save variables for future use
start_time_saving=time.time()
np.savetxt(calcdir + '/photon_number.csv', photon_number, delimiter=',')
np.savetxt(calcdir + '/photon_number_each.csv', photon_number_each, delimiter=',')
np.savetxt(calcdir + '/ele_intensity.csv', ele_intensity, delimiter=',')
end_time_saving=time.time()
print('Finished Saving files. Running Wall Time = %10.3f second' % (end_time_saving - start_time_saving))

#plot
start_time_plot=time.time()
plot_t=np.arange(0,tmax+1,dt*savestep)
#plot_t=np.insert(plot_t,0,-1)

#plot_rho=plt.figure(0)
for i in np.arange(num_atom):
    plt.plot(plot_t,avg_wavefn[:,2*i], label='c0**2, atom ' +str(i))
    plt.plot(plot_t,avg_wavefn[:,2*i+1], '-', label='c1**2, atom ' +str(i))
    plt.plot(plot_t,avg_wavefn[:,2*i]+avg_wavefn[:,2*i+1], label='c0**2+c1**2, atom ' +str(i))
plt.ylim([0, 1.05])
plt.xlabel("Time")
plt.ylabel("Atomic Population")
plt.legend()
plt.savefig(calcdir + '/Rho.png')
plt.clf()

#plot_pho=plt.figure(1)
plt.plot(plot_t,photon_number-photon_number[0], label='center')
#plt.plot(plot_t,photon_numberzp-photon_numberzp[0], label='ZP')
#plt.plot(plot_t,photon_number-photon_number[0]+photon_numberzp-photon_numberzp[0], label='all')
plt.xlabel("Time")
plt.ylabel("Photon Number")
plt.legend()
plt.savefig(calcdir + "/Photon_N.png")
plt.clf()

colors = [ cmx.jet(x) for x in np.linspace(0,1,int(2*N)) ]

#plot_pho_each=plt.figure(3)
#plt.plot(plot_t, photon_number, label='photon number')
for a in np.arange(int(2*N)):
    plt.plot(np.arange(0, tmax + 1, intens_save_t),
             photon_number_each[a,:]-photon_number_each[a,0], 'x-', label=str(alpha[a]), color=colors[a])
#    plt.plot(np.arange(0, tmax + 1, intens_save_t),
#             photon_number_eachzp[a,:]-photon_number_eachzp[a,0], 'x-', label='ZP ' + str(alpha[a]), color=colors[a])
#plt.plot(plot_t,photon_number_each[1,:] - photon_number_each[1,0], label=str(alpha[1]))
plt.xlabel("Time")
plt.ylabel("Photon Number")
plt.legend()
plt.savefig(calcdir + "/Photon_N_each.png")
plt.clf()

#plot_Es=plt.figure(2)
Ec=avg_E[:,0] - avg_E[0,0]
#Eczp=avg_E[:,1] - avg_E[0,1]
plt.plot(plot_t, Ec, '-.', label='Ec')
#plt.plot(plot_t, Eczp, '-.', label='Eczp')
Eq_Eqc_all=np.zeros_like(Ec)
for i in np.arange(num_atom):
    Eq = avg_E[:,i+1] - avg_E[0,i+1]
    Eqc = avg_E[:,i+1+num_atom] - avg_E[0,i+1+num_atom]
    plt.plot(plot_t, Eq, label='Eq' +str(i))
    plt.plot(plot_t, Eqc, label='Eqc' +str(i))
    plt.plot(plot_t, Eq + Eqc, '--', label='Eq'+str(i) + ' + Eqc' +str(i))
    Eq_Eqc_all += Eq + Eqc
plt.plot(plot_t, Ec + Eq_Eqc_all, label='System E')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energy (au)")
plt.savefig(calcdir + "/E.png")
plt.clf()

for i in np.arange(0,ele_intensity[1,:].size,1):
    plot = plt.figure(100+i, figsize=(10,3), dpi=100)
    plt.plot(r * au_to_micron, ele_intensity[:,i], label='center')
   # plt.plot(r * au_to_micron, ele_intensityzp[:, i], label='zp')
    #for j in np.arange(num_atom):
    #    plt.vlines(r_atom[j]* au_to_micron, -0.1, 0.1, colors='red')
    plt.ylim([-0.0006,0.0006])
    plt.legend()
    plt.xlabel("Cavity Distance (um)")
    plt.ylabel("Intensity (a.u.)")
    plt.text(1, 0.0005, "t=" + str(i * intens_save_t))
    plt.savefig(calcdir + "/t"+str(i*intens_save_t)+"compare.png")
    plt.clf()

for i in np.arange(photon_number_each.shape[1]):
    plot = plt.figure(1000+i)
    plt.plot(alpha, photon_number_each[:,i] - photon_number_each[:,0], label='center')
    #plt.plot(alpha, photon_number_eachzp[:, i] - photon_number_eachzp[:, 0], label='ZP')
    #plt.plot(alpha, photon_number_each[:,i] - photon_number_each[:,0]
    #         + photon_number_eachzp[:, i] - photon_number_eachzp[:, 0], label='all')
    plt.ylim([-0.02, 0.08])
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("<Nalpha>t - <Nalpha>t=0")
    plt.text(alpha[0], 0.04, "t=" + str(i * intens_save_t))
    plt.savefig(calcdir + "/Photon_t"+str(i*intens_save_t)+".png")
    plt.clf()

end_time_plot=time.time()
print('Finished Generating figures. Running Wall Time = %10.3f second' % (end_time_plot - start_time_plot))

end_time_all=time.time()
print('Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))
