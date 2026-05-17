import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import (forward_op_tomo_3d,
forward_op_tomo_3d_transpose, datacube_generator, tomomtx_gen,
gauss_pix)
from slitless.forward_legacy import (forward_op_tomo_3d_v0, forward_op_tomo_3d_k3,
forward_op_tomo_3d_transpose_k3)
from slitless.recon import tomoinv, tomoinv0, gauss_pmf_fitter2
from scipy.optimize import curve_fit

param3d = np.zeros((3,11,1))
param3d[2] = 1
param3d[:,::2,0] = np.array([1,0.5,1.3])[:,None]
param3d0 = np.ones_like(param3d)
param3d0[1] = 0

dc = datacube_generator(param3d)
M,N,R = dc.shape
orders=[0,-1,1]

print('dc shape: {}'.format(dc.shape))
H = tomomtx_gen(dc.shape[:2], orders=orders)

y = H @ dc.reshape(-1,R)

xh = np.linalg.pinv(H) @ y
# xh[xh<0] = 0

line = xh.reshape(dc.shape)
xhg = gauss_pmf_fitter2(line)
xhgr = datacube_generator(xhg)

fig, ax = plt.subplots(1,3, figsize=(15,5))
im = ax[0].imshow(dc.squeeze())
ax[0].set_title('True')
fig.colorbar(im, ax=ax[0])
im = ax[1].imshow(xh.reshape(M,N))
ax[1].set_title('Pinv')
fig.colorbar(im, ax=ax[1])
im = ax[2].imshow(xhgr.reshape(M,N))
ax[2].set_title('Pinv GaussPFMfit')
fig.colorbar(im, ax=ax[2])
plt.show()

def gauss(x,inten, vel, width):
    return inten * gauss_pix(x,vel+len(x)//2,width)

xhg2 = []

for i in range(N):
    par, var = curve_fit(gauss, np.arange(len(line)), line[:,i,0], p0=[1,0,1],
    bounds=((0,-2,1),(1,2,2.3)))
    xhg2.append(par)

xhg2 = np.array(xhg2).T[:,:,None]
xhgr2 = datacube_generator(xhg2)

fig, ax = plt.subplots(1,3, figsize=(15,5))
im = ax[0].imshow(dc.squeeze())
ax[0].set_title('True')
fig.colorbar(im, ax=ax[0])
im = ax[1].imshow(xh.reshape(M,N))
ax[1].set_title('Pinv')
fig.colorbar(im, ax=ax[1])
im = ax[2].imshow(xhgr2.reshape(M,N))
ax[2].set_title('Pinv ScipyFit')
fig.colorbar(im, ax=ax[2])
plt.show()

recon, loss = tomoinv(
    meas=y.reshape(3,N,R),
    # init_recon=datacube_generator(param3d0),
    stepsize=1e-1,
    lam=1e-1,
    proj='gauss',
    data_step='inv',
    positivity=False,
    numiter=100
    )
recon = datacube_generator(recon)

plt.figure()
plt.plot(loss)
plt.title('Losses vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(which='both', axis='both')
plt.show()

fig, ax = plt.subplots(1,3, figsize=(15,5))
im = ax[0].imshow(dc.squeeze())
ax[0].set_title('True')
fig.colorbar(im, ax=ax[0])
im = ax[1].imshow(xh.reshape(M,N))
ax[1].set_title('Pinv')
fig.colorbar(im, ax=ax[1])
im = ax[2].imshow(recon.reshape(M,N))
ax[2].set_title('ProjGD Recon')
fig.colorbar(im, ax=ax[2])
plt.show()