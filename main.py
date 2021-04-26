#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from scipy.linalg import cholesky, cho_solve, solve_triangular

# =============================================================================
# Function
# =============================================================================


def C_SE(x1, x2, a=1, b=1):
    return a * np.exp(- b * pairwise_distances(x1, x2)**2)


def C_LIN(x1, x2, a=1, b=1):
    return x1 @ x2.T


def C_NN(x1, x2, a=1, b=1):
    assert x1.shape[1] == x2.shape[1]
    b = b / x1.shape[1]
    X_11 = np.sum(x1**2, axis=1)[:, np.newaxis]
    X_12 = x1 @ x2.T
    X_22 = np.sum(x2**2, axis=1)[np.newaxis, :]
    return a * np.arcsin(2*b*X_12 / np.sqrt(1+2*b*X_11) / np.sqrt(1+2*b*X_22))


def reshape_to_3D(field_2d, nan_index, ref):
    field_3d = np.zeros((field_2d.shape[0], nan_index.size)) * np.nan
    field_3d[:, nan_index] = field_2d
    field_3d = field_3d.reshape((field_2d.shape[0],) + ref.shape[1:])
    return field_3d


def ACC(F, A):
    """Anomaly correlation coefficient"""

    C = prcp.stack(x=('lat', 'lon')).dropna('x').mean('time')
    f = F - C
    a = A - C
    f_mean = f.mean('instance')
    a_mean = a.mean('instance')

    arg = (f - f_mean) * (a - a_mean)
    arg = arg.sum('instance')

    div1 = (f - f_mean)**2
    div1 = div1.sum('instance')
    div2 = (a - a_mean)**2
    div2 = div2.sum('instance')

    return arg / xr.ufuncs.sqrt(div1 * div2)


# =============================================================================
# Pre-processing #%%
# =============================================================================

# SST
# -----------------------------------------------------------------------------
# Global SST on a 4x4 grid from 1979-2020, weekly
sst = xr.open_dataarray('./data/era5_sst_1979-2020_weekly.nc')

# weekly anomalies
weeks = sst.time.dt.isocalendar().week
sst = sst.groupby(weeks) - sst.groupby(weeks).mean()

# Precipitation
# -----------------------------------------------------------------------------
# European precipitation on a 1x1 grid from 1979-2020, weekly
prcp = xr.open_dataarray('./data/era5_prcp_1979-2020_weekly.nc')

# weekly anomalies
weeks = prcp.time.dt.isocalendar().week
prcp = prcp.groupby(weeks) - prcp.groupby(weeks).mean()

# compare SST with precipitation 4 weeks later
sst     = sst.isel(time=slice(None, -4))
prcp    = prcp.isel(time=slice(4, None))


sst.shape  # (time, lat, lon)
prcp.shape  # (time, lat, lon)

# reshape to 2D
sst_2d = sst.stack(x=('lat', 'lon'))
prcp_2d = prcp.stack(x=('lat', 'lon'))
sst_index = ~xr.ufuncs.isnan(sst_2d)[0]
prcp_index = ~xr.ufuncs.isnan(prcp_2d)[0]

sst_clean = sst_2d.dropna('x').values
prcp_clean = prcp_2d.dropna('x').values

# Split train/test set #%%
# -----------------------------------------------------------------------------
N_test = 214

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    sst_clean, prcp_clean, test_size=N_test
)
N_train = sst_clean.shape[0] - N_test


# =============================================================================
# GP Model #%%
# =============================================================================
# Training:
#   Data X dim(N, N_in)
#   Output Y dim(N, 1)
# Prediction:
#   Input (1, N_in)
#   Output (1, 1)
#
# Prior:
#   y_r = y(x_r),
#       x_r sst field at time r,
#       y_r precipitation at arb. location at time r+2week
#   y_s = y(x_s),
#       x_s sst field at time s,
#       y_s precipitation at arb. location at time s+2week
#
# Kernel:
#   <<y_r, y_s>> = C(x_r, x_s) = a exp(- \sum_i (x_ri - x_si)^2)
#
# Posterior:
#   mu_* = k_r K_rs^-1 y_s, K_rs = C(x_r, x_s), k_r = C(x_*, x_r)
#   var_* = kappa - k_r K_rs^-1 k_s, kappa = C(x_*, x_*)
#

eps = 1e-3
C = C_NN
C_kwargs = dict(a=.5, b=1e-1)

K = C(X_train, X_train, **C_kwargs)
k = C(X_train, X_test, **C_kwargs)
kappa = C(X_test, X_test, **C_kwargs)

L = cholesky(K + eps * np.eye(N_train), lower=True)
alpha = cho_solve((L, True), y_train)
mu = np.dot(k.T, alpha)
v = solve_triangular(L, k, lower=True)
var = kappa - np.dot(v.T, v)
sigma = np.sqrt(np.diag(var))


# =============================================================================
# Validation #%%
# =============================================================================


predictions = reshape_to_3D(mu, prcp_index, prcp)
observations = reshape_to_3D(y_test, prcp_index, prcp)

gp_prcp = xr.Dataset(
    data_vars=dict(
        pred=(['instance', 'lat', 'lon'], predictions),
        true=(['instance', 'lat', 'lon'], observations)
    ),
    coords=dict(
        lat=prcp.lat,
        lon=prcp.lon
    )
)


plot_instance = np.arange(0, 9)
gp_prcp.pred.sel(instance=plot_instance).plot(
    x='lon', y='lat', col='instance', figsize=(8, 7),
    col_wrap=3, vmin=-4, vmax=4, cmap='BrBG')
plt.savefig('predictions.jpg', dpi=200)
plt.show()

gp_prcp.true.sel(instance=plot_instance).plot(
    x='lon', y='lat', col='instance', figsize=(8, 7),
    col_wrap=3, vmin=-4, vmax=4, cmap='BrBG')
plt.savefig('truth.jpg', dpi=200)
plt.show()


gp_prcp_clean = gp_prcp.stack(x=('lat', 'lon')).dropna(dim='x')
acc = ACC(gp_prcp_clean.pred, gp_prcp_clean.true).unstack()

levels = [-1, -.9, -.6, -.3, -.1, .1, .3, .6, .9, 1]
acc.plot(cmap='RdYlGn', vmin=-1, vmax=1, levels=levels, figsize=(8, 6))
plt.title('Anomaly Correlation Coefficient (ACC)', loc='left')
plt.title('$n=214$', loc='right')
plt.tight_layout()
plt.savefig('acc.jpg', dpi=200)
