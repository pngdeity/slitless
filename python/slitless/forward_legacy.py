"""Deprecated forward model variants. Use slitless.forward instead."""
import warnings
warnings.warn(
    "forward_legacy is deprecated. Use slitless.forward "
    "(forward_op_tomo_3d, forward_op_tomo_3d_transpose) which accept "
    "arbitrary spectral orders.",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np


def forward_op_tomo_2d(datacube):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    M,M = datacube.shape
    dc_p = np.zeros((M, 2*M-1))
    dc_p[:,:M] = datacube
    dc_m = dc_p.copy()
    for i,r in enumerate(np.arange(M)-M//2):
        dc_p[i] = np.roll(dc_p[i], r)
        dc_m[i] = np.roll(dc_m[i], -r)
    # return dc_p, dc_m

    dc_0 = np.sum(datacube, axis=0)
    dc_m = np.sum(dc_m, axis=0)[:M]
    dc_p = np.sum(dc_p, axis=0)[:M]
    return np.stack((dc_0, dc_m, dc_p), axis=0)


# DEPRECATED: use forward_op_tomo_3d instead (still imported by scripts/)
def forward_op_tomo_3d_k3(dc, inf=False):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    M,M,N = dc.shape
    dc_p = np.zeros((M,2*M-1,N))
    dc_p[:,:M] = dc
    dc_m = dc_p.copy()
    for i,r in enumerate(np.arange(M)-M//2):
        dc_p[i] = np.roll(dc_p[i], r, axis=0)
        dc_m[i] = np.roll(dc_m[i], -r, axis=0)
    # return dc_p, dc_m

    dc_0 = np.sum(dc, axis=0)
    dc_m = np.sum(dc_m, axis=0)[:M]
    dc_p = np.sum(dc_p, axis=0)[:M]
    if inf is True:
        dc_i = np.sum(dc, axis=1)
        return np.stack((dc_0, dc_m, dc_p, dc_i), axis=0)
    else:
        return np.stack((dc_0, dc_m, dc_p), axis=0)


interp2d = np.vectorize(np.interp, signature='(m),(n),(n)->(m)')


# DEPRECATED: use forward_op_tomo_3d instead (still imported by scripts/)
def forward_op_tomo_3d_v0(dc, orders=[0,-1,1], inf=False):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    M,M,N = dc.shape
    projs = []

    dc_p = np.zeros((M,2*M-1,N))
    dc_p[:,:M] = dc
    dc_m = dc_p.copy()

    M2 = 2*M-1
    dc_p2 = np.zeros((M2,M2+M-1,N))
    dc_m2 = dc_p2.copy()

    if 2 in np.abs(orders):
        dc2 = 0.5 * interp2d(np.arange(M2), np.arange(M)*2, dc.T).T

        dc_p2[:,:M] = dc2
        dc_m2 = dc_p2.copy()

        for i,r in enumerate(np.arange(M2)-M2//2):
            if 2 in orders:
                dc_p2[i] = np.roll(dc_p2[i], r, axis=0)
            if -2 in orders:
                dc_m2[i] = np.roll(dc_m2[i], -r, axis=0)

    for i,r in enumerate(np.arange(M)-M//2):
        dc_p[i] = np.roll(dc_p[i], r, axis=0)
        dc_m[i] = np.roll(dc_m[i], -r, axis=0)
    # return dc_p, dc_m

    dc_0 = np.sum(dc, axis=0)
    dc_m = np.sum(dc_m, axis=0)[:M]
    dc_p = np.sum(dc_p, axis=0)[:M]
    dc_m2 = np.sum(dc_m2, axis=0)[:M]
    dc_p2 = np.sum(dc_p2, axis=0)[:M]

    dcs = [dc_0, dc_m, dc_p, dc_m2, dc_p2]
    ordlist = [0,-1,1,-2,2]
    inds = np.where(ordlist==np.array(orders)[:,None])[1]

    dcs2 = [dcs[ind] for ind in inds]
    if inf is True:
        dc_i = np.sum(dc, axis=1)
        return np.stack(dcs2 + [dc_i], axis=0)
    else:
        return np.stack(dcs2, axis=0)


def forward_op_tomo_2d_transpose(meas):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    _,M = meas.shape
    datacube_m = np.ones((M,2*M-1))
    datacube_p = np.ones((M,2*M-1))

    datacube_0 = np.outer(np.ones(M), meas[0])
    datacube_m[:,:M] = np.outer(np.ones(M), meas[1])
    datacube_p[:,:M] = np.outer(np.ones(M), meas[2])

    for i,r in enumerate(np.arange(M)-M//2):
        datacube_p[i] = np.roll(datacube_p[i], -r)
        datacube_m[i] = np.roll(datacube_m[i], r)

    return np.stack((datacube_0, datacube_m[:,:M], datacube_p[:,:M]), axis=0)


# DEPRECATED: use forward_op_tomo_3d_transpose instead (still imported by scripts/)
def forward_op_tomo_3d_transpose_k3(meas, inf=False, smart=True):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    _,M,N = meas.shape
    if smart is True:
        datacube_m = np.ones((M,2*M-1,N)) # (lambda,y,x)
        datacube_p = np.ones((M,2*M-1,N))
    else:
        datacube_m = np.zeros((M,2*M-1,N)) # (lambda,y,x)
        datacube_p = np.zeros((M,2*M-1,N))

    datacube_0 = np.repeat(meas[0][np.newaxis], M, axis=0)
    datacube_m[:,:M] = np.repeat(meas[1][np.newaxis], M, axis=0)
    datacube_p[:,:M] = np.repeat(meas[2][np.newaxis], M, axis=0)

    for i,r in enumerate(np.arange(M)-M//2):
        datacube_p[i] = np.roll(datacube_p[i], -r, axis=0)
        datacube_m[i] = np.roll(datacube_m[i], r, axis=0)

    if inf is True:
        datacube_i = np.repeat(meas[3][:,np.newaxis], M, axis=1)
        return np.stack((datacube_0, datacube_m[:,:M], datacube_p[:,:M], datacube_i), axis=0)
    else:
        return np.stack((datacube_0, datacube_m[:,:M], datacube_p[:,:M]), axis=0)
