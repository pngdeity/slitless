import numpy as np
import torch
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import camera, shepp_logan_phantom, cell
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.ndimage import convolve1d
from slitless.config import config

SPEED_OF_LIGHT = config.speed_of_light

def gauss(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * np.exp(-0.5*((x-mean)/sigma)**2)

def gauss_pix(x, mean, sigma):
    return 0.5 * (
        erf((x-mean+0.5)/(2**0.5*sigma)) -
        erf((x-mean-0.5)/(2**0.5*sigma))
    )

def gauss_torch(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * torch.exp(-0.5*((x-mean)/sigma)**2)

def gauss_pix_torch(x, mean, sigma):
    return 0.5 * (
        torch.erf((x-mean+0.5)/(2**0.5*sigma)) -
        torch.erf((x-mean-0.5)/(2**0.5*sigma))
    )

def forward_op(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    param3d=None,
    pixelated=False,
    mask=None,
    spectral_orders=[0,-1,1]):
    """
    Given 2d arrays of intensity, doppler, and linewidth, calculate the noise
    free measurements at the specified spectral orders.

    Notes: This implementation uses[0,-1,1] of true doppler shifts in the units of
            pixels.
        true_linewidth (ndarray): 2d array of tru50ral orders.

    returns:
        measurements (ndarray): 3d array of measurements. The first dimension
            contains the specified spectral orders with the same ordering.
    """
    gauss_func = gauss_pix if pixelated else gauss
    if param3d is not None:
        true_intensity, true_doppler, true_linewidth = param3d
    if mask is not None:
        assert true_intensity.shape == mask.shape, "Mask shape does not match detector shape"
    else:
        mask = np.ones_like(true_intensity)
    aa, bb = true_intensity.shape
    out = np.zeros((len(spectral_orders),)+(aa,bb))
    diffrange = np.arange(aa)[np.newaxis,:,np.newaxis]-np.arange(aa)[np.newaxis,np.newaxis,:]
    # assume columns of detector are independent
    for z,a in enumerate(spectral_orders):
        if a == 0:
            out[z] = mask * true_intensity.copy()
            continue
        out[z] = np.einsum(
            'kij,kj->ki',
            gauss_func(
                diffrange, 
                a*true_doppler.transpose(1,0)[:,np.newaxis,:], 
                abs(a)*true_linewidth.transpose(1,0)[:,np.newaxis,:]
            ), 
            mask.transpose(1,0) * true_intensity.transpose(1,0)
        ).transpose(1,0)
    return out

def datacube_generator(param3d, pixelated=True, lamdim=64):
    # M,N = param3d.shape[1:]
    gauss_func = gauss_pix if pixelated else gauss
    cube = gauss_func(
        np.arange(lamdim)[:,np.newaxis,np.newaxis],
        param3d[1][np.newaxis,:,:] + lamdim//2,
        param3d[2][np.newaxis,:,:]
    )
    cube *= param3d[0][np.newaxis,:,:]
    return cube

def tomomtx_gen(shape, orders=[0,-1,1]):
    M,N = shape
    cols = np.outer(np.ones(M),np.arange(N))
    rows = np.outer(np.arange(M), np.ones(N))
    mtx = []
    for order in orders:
        mtx_i=[]
        if order==0:
            for i in range(N):
                mtx_i.append((cols==i).flatten())
            mtx.append(np.array(mtx_i).astype(int))
        elif order=='inf':
            for i in range(M):
                mtx_i.append((rows==i).flatten())
            mtx.append(np.array(mtx_i).astype(int))
        elif abs(order)==1:
            map_o = cols + order * (rows-M//2)
            for i in range(N):
                mtx_i.append((map_o==i).flatten())
            mtx.append(np.array(mtx_i).astype(int))
        elif abs(order)==2:
            map_o = cols + order * (rows-M//2)
            for i in range(N):
                temp = abs(map_o-i)
                temp[temp<2] = 1-0.5*temp[temp<2]
                temp *= temp<2
                mtx_i.append(0.5*temp.flatten())
            mtx.append(np.array(mtx_i))
    return np.concatenate(mtx, axis=0)

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

def forward_op_tomo_3d(dc, orders=[0,-1,1], inf=False):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    K,M,N = dc.shape
    projs = []

    for a in orders:
        if a==0:
            projs.append(np.sum(dc, axis=0))
        else:
            if abs(a)==1:
                dcc = dc
            else:
                # 1. Base Boxcar (spectral continuity)
                kernel = np.ones(abs(a)) / abs(a)
                # 2. Fix Symmetry for Even Orders
                if abs(a) % 2 == 0:
                    # Convolve with [0.5, 0.5] to fix 0.5-pixel shift
                    kernel = np.convolve(kernel, [0.5, 0.5])
                dcc = convolve1d(dc, kernel, axis=1, mode='constant')
            pad_len = abs(a)*(K-1)
            dc_p = np.zeros((K, M + pad_len, N))
            dc_p[:,:M] = dcc
            for i,r in enumerate(np.arange(K)-K//2):
                dc_p[i] = np.roll(dc_p[i], a*r, axis=0)
            projs.append(np.sum(dc_p, axis=0)[:M])
    
    if inf is True:
        dc_i = np.sum(dc, axis=1)
        return np.array(projs), dc_i
    else:
        return np.array(projs)

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

def forward_op_tomo_3d_transpose(meas, orders=[0,-1,1], inf=False):
    # axis 0 is lambda (index up -> lambda up), axis 1 is dispersion direction
    ordlist = [0,-1,1,-2,2]
    inds = np.where(ordlist==np.array(orders)[:,None])[1]

    _,M,N = meas.shape
    dc_m = np.ones((M,2*M-1,N)) # (lambda,y,x)
    dc_p = np.ones((M,2*M-1,N))
    dc_m2 = np.ones((M,3*M-2,N))
    dc_p2 = np.ones((M,3*M-2,N))

    dc_0 = np.repeat(meas[0][np.newaxis], M, axis=0)
    if -1 in orders:
        dc_m[:,:M] = np.repeat(meas[1][np.newaxis], M, axis=0)
    if 1 in orders:
        dc_p[:,:M] = np.repeat(meas[2][np.newaxis], M, axis=0)
    if -2 in orders:
        dc_m2[:,:M] = np.repeat(meas[3][np.newaxis], M, axis=0)
    if 2 in orders:
        dc_p2[:,:M] = np.repeat(meas[4][np.newaxis], M, axis=0)

    for i,r in enumerate(np.arange(M)-M//2):
        if -1 in orders:
            dc_m[i] = np.roll(dc_m[i], r, axis=0)
        if 1 in orders:
            dc_p[i] = np.roll(dc_p[i], -r, axis=0)
        if -2 in orders:
            dc_m2[i] = np.roll(dc_m2[i], 2*r, axis=0)
        if 2 in orders:
            dc_p2[i] = np.roll(dc_p2[i], -2*r, axis=0)

    dcs = [dc_0, dc_m, dc_p, dc_m2, dc_p2]
    dcs2 = [dcs[ind][:,:M] for ind in inds]
    
    if inf is True:
        dc_i = np.repeat(meas[-1][:,np.newaxis], M, axis=1)
        return np.stack(dcs2 + [dc_i], axis=0)
    else:
        return np.stack(dcs2, axis=0)

def forward_op_torch(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    pixelated=False,
    spectral_orders=[0,-1,1],
    mask=None,
    device=None):
    """
    Given 2d (or 3d where the first dimension is batch dim) arrays of intensity,
    doppler, and linewidth, calculate the noise free measurements at the
    specified spectral orders.

    Args:
        true_intensity (Tensor): 2d or 3d array of true intensities.
        true_doppler (Tensor): 2d or 3d array of true doppler shifts in the
            units of pixels.
        true_linewidth (Tensor): 2d or 3d array of true line widths in the units
            of pixels.
        pixelated (bool): if True, take the integral of Gaussian along a pixel
            instead of impulse sampling at the midpoint.
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (Tensor): 3d or 4d array of measurements. The first
        dimension is the batch dim (optional in case input arrays are 2d), and
        the second dim contains the specified spectral orders with the same
        ordering.
    """
    gauss_func = gauss_pix_torch if pixelated else gauss_torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == None:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = true_intensity.device
    if len(true_intensity.shape) == 2:
        true_intensity = true_intensity[None]
        true_doppler = true_doppler[None]
        true_linewidth = true_linewidth[None]
        if mask is not None:
            mask = mask[None]
    else:
        if mask is not None:
            mask = np.repeat(mask[np.newaxis,:,:], len(true_intensity), axis=0)
        else:
            mask = torch.ones_like(true_intensity)

    if type(mask)!=type(true_intensity):
        mask = torch.from_numpy(mask).to(device=device, dtype=torch.float)

    k, aa, bb = true_intensity.shape
    out = torch.zeros((k,) + (len(spectral_orders),)+(aa,bb))
    out = out.to(device=device, dtype=torch.float)
    diffrange = torch.arange(aa)[None,None,:,None]-torch.arange(aa)[None,None,None,:]
    diffrange = diffrange.to(device=device, dtype=torch.float)
    # assume columns of detector are independent
    for z,a in enumerate(spectral_orders):
        a = torch.Tensor([a]).to(device=device, dtype=torch.float)
        if a == 0:
            out[:,z] = mask*true_intensity.clone()
            continue
        out[:,z] = torch.einsum(
            'lkij,lkj->lki',
            gauss_func(
                diffrange, 
                a*true_doppler.permute(0,2,1)[:,:,None,:], 
                torch.abs(a)*true_linewidth.permute(0,2,1)[:,:,None,:]
            ), 
            mask.permute(0,2,1) * true_intensity.permute(0,2,1)
        ).permute(0,2,1)
    if len(true_intensity.shape) == 2:
        return out[0]
    else:
        return out

def obj_ls(x, meas=None):
    aa, bb = meas.shape[1:]
    intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
    diff = forward_op(intensity, doppler, linewidth) - meas
    return np.sum(diff**2)

class Source():
    """
    A class for holding the source parameters.

    Args:
        param3d (ndarray): 3d array of the stacked (inten, vel, width) arrays.
        inten (ndarray): 2d array of intensity values.
        vel (ndarray): 2d array of Doppler velocity values. Expects either the
            units of pixels with the pix=True, or [km/s] with pix=False argument.
        width (ndarray): 2d array of line width values. Expects either the units
            of pixels with the pix=True, or [A] with pix=False argument.
        wavelength (float): wavelenght of interest in [A].
        pix (bool): set to True if the input vel & width are given in pixel
        units; False otherwise.
    
    Attributes:
        inten
        vel
        width
        wavelength
        pix
        param3d
    """
    def __init__(
        self,
        *,
        param3d=None,
        inten=None,
        vel=None,
        width=None,
        rest_wavelength=195.117937907451,
        pix=False
    ):
        self.rest_wavelength = rest_wavelength
        if param3d is not None:
            self.param3d = param3d
            self.inten = param3d[...,0,:,:]
            self.vel = param3d[...,1,:,:]
            self.width = param3d[...,2,:,:]
        else:
            self.inten = inten
            self.vel = vel
            self.width = width
            stack = torch.stack if type(self.inten)==torch.Tensor else np.stack
            self.param3d = stack((inten, vel, width))
        self.pix = pix
    def plot(self, title='', idx=0, ssims=None, rmses=None, psnrs=None):
        if type(self.inten)==torch.Tensor:
            inten, vel, width = self.inten.cpu().numpy(), self.vel.cpu().numpy(), self.width.cpu().numpy()
        else:
            inten, vel, width = self.inten, self.vel, self.width
        if len(inten.shape) == 3:
            inten, vel, width = inten[idx], vel[idx], width[idx]
        str_int = 'Intensity'
        str_vel = 'Velocity [pix]' if self.pix else 'Velocity [km/s]'
        str_width = 'Linewidth [pix]' if self.pix else 'Linewidth [A]'
        for name, field in zip(('SSIM', 'RMSE', 'PSNR'),(ssims, rmses, psnrs)):
            if field is None:
                continue
            str_int += '\n {}: {:.3f}'.format(name, field[0])
            str_vel += '\n {}: {:.3f}'.format(name, field[1])
            str_width += '\n {}: {:.3f}'.format(name, field[2])
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        plt.suptitle(title)
        i0=ax[0].imshow(inten, cmap='hot')
        ax[0].set_title(str_int)
        fig.colorbar(i0, ax=ax[0])
        i1=ax[1].imshow(vel, cmap='seismic')
        fig.colorbar(i1, ax=ax[1])
        ax[1].set_title(str_vel)
        i2=ax[2].imshow(width, cmap='plasma')
        fig.colorbar(i2, ax=ax[2])
        ax[2].set_title(str_width)
        plt.tight_layout()
        if matplotlib.get_backend().lower() != 'agg':
            plt.show()
        return fig, ax

class Imager():
    """
    The class to hold the imager-specific parameters and measurement functions.

    Args:
        pixel_size (float): detector pixel size in um (micrometers).
        dispersion (float): dispersion of the imager in um/mA.
        dispersion_scale (float): dispersion scale of the imager in mA/pixels. 
            Has a precedence over `dispersion` if both are specified.
        instrument_psf (ndarray): 2d array of the instrument psf to be convolved
            with the measurements. [Not Implemented]
        spectral_orders (list): list of integers specifying the spectral orders
            of the measurements. this also determines the number of measurements
        pixelated (bool): if True, when simulating the measurements take the 
            integral of Gaussian along a pixel instead of impulse sampling at 
            the midpoint.

    Attributes:
        pixel_size [um]
        dispersion [um/mA]
        dispersion_scale [mA/pixels]
        spectral_orders
        srpix (Source): a Source instance in pixel units.
        meas3d (dict): dictionary holding the 2d measurement arrays with the 
            keys corresponding to spectral orders
        meas3dar (ndarray): 3d array of the measurements, whose order is the 
            same as the ordering in the `spectral_order` list.
    """
    def __init__(
        self,
        *,
        pixel_size=13.5, # um/pixel
        dispersion=1/1.65, # um/mA
        dispersion_scale=None, # A/pixel
        mid_wavelength=195.119, # A - wavelength of the mid-point of the detector
        instrument_psf=None,
        spectral_orders=[0,-1,1],
        pixelated=False,
        mask=None,
        dbsnr=None,
        max_count=None,
        avg_count=None,
        noise_model=None,
        meas=None,
        intenscale=1
    ):
        self.pixel_size = pixel_size
        self.mid_wavelength = mid_wavelength
        self.dispersion_scale = pixel_size / dispersion / 1e3
        if dispersion_scale is not None:
            self.dispersion_scale = dispersion_scale
            self.dispersion = pixel_size / dispersion_scale / 1e3
        self.spectral_orders = spectral_orders
        self.pixelated = pixelated
        self.mask = mask
        self.dbsnr = dbsnr
        self.max_count = max_count
        self.avg_count = avg_count
        self.noise_model = noise_model
        if meas is not None:
            self.meas3dar = meas
            self.meas3dar_nn = meas
        self.intenscale=intenscale

    def topix(self, source):
        """
        Takes as input a Source object which has the physical units of 
        velocity and line width, and creates another Source object as an attribute
        of the Imager, which has these parameters in the pixel units.
        """
        assert source.pix == False, "Source object is already in pixel dimensions"

        rest_cen = getattr(source, 'rest_wavelength', 195.117937907451)
        # Absolute wavelength of the shifted line
        actual_wave = rest_cen * (1 + source.vel / SPEED_OF_LIGHT)
        # Pixel shift relative to the array's central wavelength
        vel_pix = (actual_wave - getattr(self, 'mid_wavelength', 195.119)) / self.dispersion_scale
        self.srpix = Source(
            inten=source.inten/self.intenscale,
            vel=vel_pix,
            width=source.width/self.dispersion_scale,
            rest_wavelength=rest_cen,
            pix=True
        )
        return self.srpix

    def frompix(self, source, width_unit='A', array=False):
        """
        Takes as input a Source object which has the pixel units of 
        velocity and line width, and creates another Source object as an attribute
        of the Imager, which has these parameters in the physical units.
        """
        if array==False:
            assert source.pix == True, "Source object is already in physical dimensions"
            rest_cen = getattr(source, 'rest_wavelength', 195.117937907451)
            # Wavelength of the shifted line based on pixel position
            actual_wave = getattr(self, 'mid_wavelength', 195.119) + source.vel * self.dispersion_scale
            # Velocity relative to the rest wavelength
            vel_phy = SPEED_OF_LIGHT * (actual_wave - rest_cen) / rest_cen
            
            width_phy = source.width * self.dispersion_scale
            if width_unit == 'km/s':
                width_phy *= SPEED_OF_LIGHT / rest_cen
            self.srphy = Source(
                inten=source.inten*self.intenscale,
                vel=vel_phy,
                width=width_phy,
                rest_wavelength=rest_cen,
                pix=False
            )
            return self.srphy
        else:
            out = source.clone() if type(source)==torch.Tensor else source.copy()
            out[...,0,:,:]*=self.intenscale
            
            wave_cen = getattr(self, 'mid_wavelength', 195.119)
            rest_cen = getattr(self.srpix, 'rest_wavelength', 195.117937907451) if hasattr(self, 'srpix') else 195.117937907451
            
            actual_wavelengths = wave_cen + out[...,1,:,:] * self.dispersion_scale
            out[...,1,:,:] = SPEED_OF_LIGHT * (actual_wavelengths - rest_cen) / rest_cen
            
            out[...,2,:,:] *= self.dispersion_scale
            if width_unit=='km/s':
                out[...,2,:,:] *= SPEED_OF_LIGHT / rest_cen
                
            return out

    def forward_op(self, inten, vel, width):

        fwd_op = forward_op_torch if type(inten)==torch.Tensor else forward_op

        return fwd_op(
            true_intensity=inten,
            true_doppler=vel,
            true_linewidth=width,
            spectral_orders=self.spectral_orders,
            pixelated=self.pixelated,
            mask=self.mask
        )

    def get_measurements(
        self,
        sources=None,
        tomo=False,
        dbsnr=None,
        max_count=None,
        avg_count=None,
        noise_model=None,
        no_noise=None
    ):
        """
        Given a Source object, simulate and save measurements as an attribute 
        of the Imager object using the forward_op function.

        Args:
            sources (Source): a Source instance holding the source parameters
        """
        if sources==None:
            sources = self.srpix
        if sources.pix == False:
            self.topix(sources)
        else:
            self.srpix = sources
        
        if dbsnr is None:
            dbsnr = self.dbsnr
        if max_count is None:
            max_count = self.max_count
        if avg_count is None:
            avg_count = self.avg_count
        if noise_model is None:
            noise_model = self.noise_model

        if tomo is True:
            fwd_op = forward_op_tomo_3d
            self.datacube = datacube_generator(self.srpix.param3d, lamdim=21)
            self.meas3dar = fwd_op(
                self.datacube, orders=self.spectral_orders
            )
        else:
            fwd_op = forward_op_torch if type(sources.inten)==torch.Tensor else forward_op

            self.meas3dar = fwd_op(
                true_intensity=self.srpix.inten,
                true_doppler=self.srpix.vel,
                true_linewidth=self.srpix.width,
                spectral_orders=self.spectral_orders,
                pixelated=self.pixelated,
                mask=self.mask
            )

        self.meas3dar_nn = self.meas3dar.clone() if type(sources.inten)==torch.Tensor else self.meas3dar.copy()
        self.meas3dar = add_noise(
            self.meas3dar, dbsnr=dbsnr, max_count=max_count, avg_count=avg_count, 
            noise_model=noise_model, no_noise=no_noise
        )
        
        return self.meas3dar

    def plot(self, title='', noise=True):
        fig, ax = plt.subplots(1,len(self.spectral_orders), figsize=(15,5))
        plt.suptitle(title)
        for i,a in enumerate(self.spectral_orders):
            if (noise==False) and hasattr(self, 'meas3dar_nn'):
                im=ax[i].imshow(self.meas3dar_nn[i], cmap='hot')
            else:
                im=ax[i].imshow(self.meas3dar[i], cmap='hot')
            ax[i].set_title('Order {}'.format(a))
            fig.colorbar(im, ax=ax[i])
        plt.tight_layout()
        plt.show()

def add_noise(signal, dbsnr=None, max_count=None, avg_count=None, noise_model='Gaussian', no_noise=False):
    """
    Add noise to the given signal at the specified level.

    Args:
        signal (ndarray): noise-free input signal
        dbsnr (float): signal to noise ratio in dB: for Gaussian noise model, it is
        defined as the ratio of variance of the input signal to the variance of
        the noise. For Poisson model, it is taken as the average snr where snr
        of a pixel is given by the square root of its value.
        max_count (int): Max number of photon counts in the given signal
        avg_count (int): Average number of photon counts in the given signal
        noise_model (string): String that specifies the noise model. The 2 options are
        `Gaussian` and `Poisson`
        no_noise (bool): (default=False) If True, return the clean signal

    Returns:
        ndarray that is the noisy version of the input
    """
    if no_noise is True or noise_model is None:
        return signal
    assert noise_model.lower() in ('gaussian', 'poisson'), "invalid noise model"
    if noise_model.lower() == 'poisson' and max_count is None and avg_count is None:
        avg_count = dbsnr**2

    sig = signal.cpu().numpy() if type(signal)==torch.Tensor else signal

    if noise_model.lower() == 'gaussian':
        var_sig = np.var(sig, axis=(-1,-2), keepdims=True)
        std_noise = (var_sig / 10**(dbsnr / 10))**0.5
        out = np.random.normal(sig, std_noise)
    elif noise_model.lower() == 'poisson':
        if max_count is not None:
            scalar = max_count / np.max(sig, axis=(-1,-2), keepdims=True)
            sig_scaled = sig * scalar
            # print('SNR:{}'.format(np.sqrt(sig_scaled.mean())))
            out = np.random.poisson(sig_scaled) / scalar
        elif avg_count is not None:
            scalar = avg_count / np.mean(sig, axis=(-1,-2), keepdims=True)
            sig_scaled = sig * scalar
            out = np.random.poisson(sig_scaled) / scalar
        else:
            avg_brightness = 10**(dbsnr / 20)**2
            scalar = avg_brightness / sig.mean(axis=(-1,-2), keepdims=True)
            sig_scaled = sig * scalar
            out = np.random.poisson(sig_scaled) / scalar

    if type(signal)==torch.Tensor:
        out = torch.from_numpy(out).to(device=signal.device, dtype=signal.dtype)

    return out