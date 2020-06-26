import numpy as np
from scipy import sparse
from .set_operations import w_subset
from .util import lat2W, lat2SW
from .weights import W, WSP
from warnings import warn

try:
    import pandas as pd
    from xarray import DataArray
except ImportError:
    raise ImportError(
        "xarray and pandas must be installed to use raster related functionality")

__all__ = ['da_checker', 'da2W', 'da2SW', 'toDataArray']


def da_checker(da, band):
    """
    xarray dataarray checker
    Parameters
    ----------
    da      : xarray.DataArray
              raster file accessed using xarray.open_rasterio method
    band    : int
              user selected band
    Returns
    -------
    band    : int
              return default band value
    """
    if not isinstance(da, DataArray):
        raise TypeError("da must be an instance of xarray.DataArray")
    else:
        if band is None:
            if da.sizes['band'] != 1:
                warn(f'Multiple bands detected in the DataArray. Using default band 1 for further computation')
            band = 1
    return band


def mask_sw_row(sw, mask, ids):
    """
    Set the rows to zero of the csr matrix

    Parameters
    ----------
    sw         : sparse.csr_matrix
                 instance of sparse weight matrix 
    mask       : boolean array
                 mask array.
    ids        : 1D numpy array
                 containing indices of rows to be zeroed 
    Returns
    -------
    sw    : scipy.sparse.csr_matrix
           instance of a scipy sparse matrix      
    """
    if not isinstance(sw, sparse.csr_matrix):
        sw.tocsr()
    nnz_per_row = np.diff(sw.indptr)
    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[ids] = 0
    sw.data = sw.data[mask]
    sw.indices = sw.indices[mask]
    sw.indptr[1:] = np.cumsum(nnz_per_row)
    return sw


def da2W(da, criterion, **kwargs):
    """
    Create a W object from rasters(xarray.DataArray)

    Parameters
    ----------
    da         : xarray.DataArray
                 raster file accessed using xarray.open_rasterio method
    criterion  : {"rook", "queen"}
                 option for which kind of contiguity to build
    Returns
    -------
    w    : libpysal.weights.W
           instance of spatial weights class W
    """
    if criterion is not 'rook':
        rook = False
    else:
        rook = True
    w = lat2W(*da[0].shape, rook=rook, **kwargs)
    ser = da.to_series()
    id_order = np.where(ser != da.nodatavals[0])[0]
    w = w_subset(w, id_order)
    ser = ser[ser != da.nodatavals[0]]
    w.ids = ser.index
    return w


def da2SW(da, criterion, **kwargs):
    """
    Generate a sparse W matrix from rasters(xarray.DataArray)

    Parameters
    ----------
    da         : xarray.DataArray
                 raster file accessed using xarray.open_rasterio method
    criterion  : {"rook", "queen"}
                 option for which kind of contiguity to build
    Returns
    -------
    sw    : libpysal.weights.WSP
           instance of spatial weights class WSP
    """
    sw = lat2SW(*da[0].shape, criterion=criterion)
    ser = da.to_series()
    id_order = np.where(ser == da.nodatavals[0])[0]
    mask = np.ones((sw.shape[0],), dtype=np.bool)
    mask[id_order] = False
    sw = mask_sw_row(sw, mask, id_order)
    sw = sw.transpose()
    sw = mask_sw_row(sw, mask, id_order)
    sw = sw.transpose()
    sw = WSP(sw)
    print(list(np.where(ser != da.nodatavals[0])[0]))
    return sw


def toDataArray(data, id_order, indices, attrs):
    """
    converts calculated results to a DataArray 

    Arguments
    ---------
    data    :   array
                data values stored in 1d array
    id_order:   list/array
                Ordered sequence of IDs of weight object
    indices :   Dictionary/xarray.core.coordinates.DataArrayCoordinates
                coordinates from original DataArray
    attrs   :   Dictionary
                attributes from original DataArray 
    Returns
    -------
    
    da : xarray.DataArray
         instance of xarray.DataArray
    """
    coords = {}
    for key,value in indices.items():
        if key == "band":
            coords[key] = np.asarray([1])
        else:
            coords[key] = value
    ser = pd.Series(data, index=id_order)
    shape = tuple(len(value) for value in coords.values())
    n = shape[1]*shape[2]
    ser = ser.reindex(index=range(n), fill_value=attrs['nodatavals'][0])
    data = ser.to_numpy().reshape((shape))
    da = DataArray(data=data, dims=tuple(key for key in coords), coords=coords, attrs=attrs)
    return da
