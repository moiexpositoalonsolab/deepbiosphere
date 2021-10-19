"""Copy valid pixels from input files to an output file."""
import os
from contextlib import contextmanager
import logging
import math
from pathlib import Path
import warnings

import numpy as np

import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.enums import Resampling
from rasterio import windows
from deepbiosphere.scripts import new_window
from rasterio.transform import Affine


logger = logging.getLogger(__name__)

# deciding that this isn't important in the meantime, focus on finishing
def copy_average(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
# so my goal is to take values that are in merged_data
# and are legal in new_data and average them?
    np.logical_and(np.logical_not(new_mask),np.logical_not(merged_mask)) # the average mask
    np.logical_not(new_mask, out=mask)# get places where data to read in is not nan
    np.logical_and(merged_mask, mask, out=mask) # get where data has already been read in
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")

    mask = np.logical_and(np.logical_not(new_mask),np.logical_not(merged_mask)) # the average mask
    # the value to add (the average)
    avg_data = ((merged_data[avg_mask]+new_data[avg_mask])/2).astype(np.uint8)
    np.copyto(avg_data, new_data, where=mask, casting="unsafe")

def copy_first(merged_data, new_data, merged_mask, new_mask, **kwargs):
# put data where there hasn't been data added yet
    mask = np.empty_like(merged_mask, dtype="bool")
# and be sure to not put data that's nan from the raster
    np.logical_not(new_mask, out=mask)
# if no data has been put yet and it's not nan, add it!
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_last(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_not(new_mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_min(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.minimum(merged_data, new_data, out=merged_data, where=mask)
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_max(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.maximum(merged_data, new_data, out=merged_data, where=mask)
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


MERGE_METHODS = {
    'first': copy_first,
    'last': copy_last,
    'min': copy_min,
    'max': copy_max
}



def new_merge(
    datasets,
    bounds=None,
    res=None,
    nodata=None,
    dtype=None,
    precision=None,
    indexes=None,
    output_count=None,
    resampling=Resampling.nearest,
    method="first",
    target_aligned_pixels=False,
    dst_path=None,
    dst_kwds=None,
):
    """Copy valid pixels from input files to an output file.

    All files must have the same number of bands, data type, and
    coordinate reference system.

    Input files are merged in their listed order using the reverse
    painter's algorithm (default) or another method. If the output file exists,
    its values will be overwritten by input values.

    Geospatial bounds and resolution of a new output file in the
    units of the input file coordinate reference system may be provided
    and are otherwise taken from the first input file.

    Parameters
    ----------
    datasets : list of dataset objects opened in 'r' mode, filenames or pathlib.Path objects
        source datasets to be merged.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input rasters.
    res: tuple, optional
        Output resolution in units of coordinate reference system. If not set,
        the resolution of the first raster is used. If a single value is passed,
        output pixels will be square.
    nodata: float, optional
        nodata value to use in output file. If not set, uses the nodata value
        in the first input raster.
    dtype: numpy dtype or string
        dtype to use in outputfile. If not set, uses the dtype value in the
        first input raster.
    precision: float, optional
        Number of decimal points of precision when computing inverse transform.
    indexes : list of ints or a single int, optional
        bands to read and merge
    output_count: int, optional
        If using callable it may be useful to have additional bands in the output
        in addition to the indexes specified for read
    resampling : Resampling, optional
        Resampling algorithm used when reading input files.
        Default: `Resampling.nearest`.
    method : str or callable
        pre-defined method:
            first: reverse painting
            last: paint valid new on top of existing
            min: pixel-wise min of existing and new
            max: pixel-wise max of existing and new
        or custom callable with signature:

        def function(merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None):

            Parameters
            ----------
            merged_data : array_like
                array to update with new_data
            new_data : array_like
                data to merge
                same shape as merged_data
            merged_mask, new_mask : array_like
                boolean masks where merged/new data pixels are invalid
                same shape as merged_data
            index: int
                index of the current dataset within the merged dataset collection
            roff: int
                row offset in base array
            coff: int
                column offset in base array

    target_aligned_pixels : bool, optional
        Whether to adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``
        options of GDAL utilities.  Default: False.
    dst_path : str or Pathlike, optional
        Path of output dataset
    dst_kwds : dict, optional
        Dictionary of creation options and other paramters that will be
        overlaid on the profile of the output dataset.

    Returns
    -------
    tuple

        Two elements:

            dest: numpy ndarray
                Contents of all input rasters in single array

            out_transform: affine.Affine()
                Information for mapping pixel coordinates in `dest` to another
                coordinate system

    """
    if method in MERGE_METHODS:
        copyto = MERGE_METHODS[method]
    elif callable(method):
        copyto = method
    else:
        raise ValueError('Unknown method {0}, must be one of {1} or callable'
                         .format(method, list(MERGE_METHODS.keys())))

    # Create a dataset_opener object to use in several places in this function.
    if isinstance(datasets[0], str) or isinstance(datasets[0], Path):
        dataset_opener = rasterio.open
    else:

        @contextmanager
        def nullcontext(obj):
            try:
                yield obj
            finally:
                pass

        dataset_opener = nullcontext

    with dataset_opener(datasets[0]) as first:
        first_profile = first.profile
        first_res = first.res
        nodataval = first.nodatavals[0]
        dt = first.dtypes[0]

        if indexes is None:
            src_count = first.count
        elif isinstance(indexes, int):
            src_count = indexes
        else:
            src_count = len(indexes)

        try:
            first_colormap = first.colormap(1)
        except ValueError:
            first_colormap = None

    if not output_count:
        output_count = src_count

    # Extent from option or extent of all inputs
    if bounds:
        dst_w, dst_s, dst_e, dst_n = bounds
    else:
        # scan input files
        xs = []
        ys = []
        for dataset in datasets:
            with dataset_opener(dataset) as src:
                left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    # Resolution/pixel size
    if not res:
        res = first_res
    elif not np.iterable(res):
        res = (res, res)
    elif len(res) == 1:
        res = (res[0], res[0])

    if target_aligned_pixels:
        dst_w = math.floor(dst_w / res[0]) * res[0]
        dst_e = math.ceil(dst_e / res[0]) * res[0]
        dst_s = math.floor(dst_s / res[1]) * res[1]
        dst_n = math.ceil(dst_n / res[1]) * res[1]

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
# because round() has weird behavior, instead we're going to
# round up the size of the array always and slightly
# stretch the rasters to fill it
    output_width = math.ceil((dst_e - dst_w) / res[0])
    output_height = math.ceil((dst_n - dst_s) / res[1])
    output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])

    if dtype is not None:
        dt = dtype
        logger.debug("Set dtype: %s", dt)

    out_profile = first_profile
    out_profile.update(**(dst_kwds or {}))

    out_profile["transform"] = output_transform
    out_profile["height"] = output_height
    out_profile["width"] = output_width
    out_profile["count"] = output_count
    if nodata is not None:
        out_profile["nodata"] = nodata

    # create destination array
    dest = np.zeros((output_count, output_height, output_width), dtype=dt)
    if nodata is not None:
        nodataval = nodata
        logger.debug("Set nodataval: %r", nodataval)

    if nodataval is not None:
        # Only fill if the nodataval is within dtype's range
        inrange = False
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            inrange = (info.min <= nodataval <= info.max)
        elif np.issubdtype(dt, np.floating):
            if math.isnan(nodataval):
                inrange = True
            else:
                info = np.finfo(dt)
                inrange = (info.min <= nodataval <= info.max)
        if inrange:
            dest.fill(nodataval)
        else:
            warnings.warn(
                "The nodata value, %s, is beyond the valid "
                "range of the chosen data type, %s. Consider overriding it "
                "using the --nodata option for better results." % (
                    nodataval, dt))
    else:
        nodataval = 0
    for idx, dataset in enumerate(datasets):
        with dataset_opener(dataset) as src:
            # Real World (tm) use of boundless reads.
            # This approach uses the maximum amount of memory to solve the
            # problem. Making it more efficient is a TODO.

            if disjoint_bounds((dst_w, dst_s, dst_e, dst_n), src.bounds):
                logger.debug("Skipping source: src=%r, window=%r", src)
                continue
# 1. Compute spatial intersection of destination and source
            src_w, src_s, src_e, src_n = src.bounds
            # so src.bounds is the extent of the raster
            # in the coordinate system of the raster (ie: lat/lon degree extent)
            # so, what this below chunk says is that if the current raster is bigger than the bounds outlined
# is outside of your bounds, only take to the b ounds
            int_w = src_w if src_w > dst_w else dst_w
            int_s = src_s if src_s > dst_s else dst_s
            int_e = src_e if src_e < dst_e else dst_e
            int_n = src_n if src_n < dst_n else dst_n
            # so then this next section just takes these extents (ignoring the boudns ig?)
            # plus the affine transform and tries to line up the pixels
            # 2. Compute the source window
            # because we wanna be safe and avoid padding errors
# we take the floor of the min coordinate and the ceil of the max
# coordinate so that the downsampling of the raster meets the
# largest bounding box it can fit. This may lead to some overlap
# on the edges, and currently we're using painter's overlap to
# deal with this inconsistency. It's such a boundary thing that it
# shouldn't be an issue though...
            src_window = new_window.new_from_bounds(
                int_w, int_s, int_e, int_n, src.transform, precision=precision
            )

            # 3. Compute the destination window
            dst_window = new_window.new_from_bounds(
                int_w, int_s, int_e, int_n, output_transform, precision=precision
            )
            # 4. Read data in source window into temp
# no longer convinced all this junk is relevant below
# kinda problematic anyway I don't think it's relevant
# basically this code rounds ah and there may be offending
# problems with the rounding as well, rip
# yeah let's avoid like the plague...
            temp_shape = (src_count, dst_window.height, dst_window.width)
            temp_src = src.read(
                out_shape=temp_shape,
                window=src_window,
                boundless=False,
                masked=True,
                indexes=indexes,
                resampling=resampling,
            )


            # 5. Copy elements of temp into dest
            region = dest[:, dst_window.row_off : dst_window.row_off + dst_window.height, dst_window.col_off : dst_window.col_off+ dst_window.width]

            # okay so this region mask will select for regions that have nooo data fill so far
            if math.isnan(nodataval):
                region_mask = np.isnan(region)
            elif np.issubdtype(region.dtype, np.floating):
                region_mask = np.isclose(region, nodataval)
            else:
                region_mask = region == nodataval

        temp_mask = np.ma.getmask(temp_src)
      #  so, copyto is a special case of np.copyto I think
      #  that either takes the current or previous or min or max
      #  value of that set. I don't know what the default is
        copyto(region, temp_src, region_mask, temp_mask, index=idx, roff=dst_window.row_off, coff=dst_window.col_off)

    if dst_path is None:
        return dest, output_transform

    else:
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(dest)
            if first_colormap:
                dst.write_colormap(1, first_colormap)
