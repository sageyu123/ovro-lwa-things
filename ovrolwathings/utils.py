import json
import os
import re
from datetime import datetime
from glob import glob

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from astropy.time import Time
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, MinuteLocator
from scipy.interpolate import interp1d


def create_pvimgdata(data, spacing, origin, dimensions):
    """
    Create an ImageData object.

    :param data: np.ndarray
        Image data.
    :param spacing: tuple
        Spacing of the image data.
    :param origin: tuple
        Origin of the image data.
    :param dimensions: tuple
        Dimensions of the image data.
    :return: pv.ImageData
        PyVista ImageData object.
    """
    img_data = pv.ImageData()
    img_data.spacing = spacing
    img_data.origin = origin
    img_data.dimensions = dimensions
    img_data['scalar'] = data
    return img_data


def extract_reffreq(meta):
    """
    Extract the reference frequency value from the meta dictionary.

    Parameters:
    meta (dict): The metadata dictionary containing WCS information.

    Returns:
    float: The reference frequency value, or None if not found.
    """
    # Find the frequency axis key
    for key in meta:
        if key.lower().startswith('ctype') and meta[key].lower() == 'freq':
            axis_num = key[-1]  # Assumes keys are in the format CTYPE1, CTYPE2, etc.
            crval_key = f'CRVAL{axis_num}'
            cunits_key = f'CUNIT{axis_num}'
            try:
                return float(meta[crval_key]) * u.Unit(meta[cunits_key])
            except KeyError:
                print(f"Corresponding {crval_key} not found in metadata")
                return None

    # If no frequency axis is found
    print("Frequency axis information not found in metadata")
    return None


def shift2pxy(shiftx, shifty, freq):
    """
    Calculate refraction correction coefficients from the given shifts in pixel unit and frequency.

    :param shiftx: float
        Solar X shift in pixel.
    :param shifty: float
        Solar Y shift in pixel.
    :param freq: float
        Frequency of the image in MHz.
    :return: tuple
        Computed refraction correction coefficients (px, py).
    """
    freqhz = np.array(freq) * 1e6
    px = shiftx * freqhz ** 2
    py = shifty * freqhz ** 2
    return px, py


def pxy2shifts(px, py, freqs):
    """
    Convert refraction correction coefficients to shifts in pixel units.

    :param px: float
        Refraction correction coefficient in x direction.
    :param py: float
        Refraction correction coefficient in y direction.
    :param freqs: list
        List of frequencies in MHz.
    :return: tuple
        Solar X and Y shifts (shifts_x, shifts_y) in pixel units.
    """
    freqshz = np.array(freqs) * 1e6
    shifts_x = px[0] * 1 / freqshz ** 2 + px[1]
    shifts_y = py[0] * 1 / freqshz ** 2 + py[1]
    return shifts_x, shifts_y


def read_rfrcorr_parms_json(filename):
    with open(filename, 'r') as file:
        params = json.load(file)
        px = np.array([params['px0'], params['px1']])
        py = np.array([params['py0'], params['py1']])
        if 'time' in params:
            time = params['time']
        else:
            time = None
        return px, py, time


def plot_interpolated_params(tims, px_values, py_values, target_times, px_at_targets, py_at_targets, save_fig=True,
                             fig_parms=None, ax_parms=None, workdir='./', tim=None, params=None):
    if fig_parms is None:
        fig_parms, ax_parms = plt.subplots(4, 1, figsize=(4, 7), sharex=True)
    timplt = Time(tims).plot_date
    target_times_plt = Time(target_times).plot_date

    ax_parms[0].plot(timplt, px_values[:, 0], marker='o', ls='', color='tab:blue', label='data')
    ax_parms[0].plot(target_times_plt, px_at_targets[:, 0], '-', color='tab:blue', label='fit')
    ax_parms[0].set_ylabel('σₓ [arcsec/Hz²]')

    ax_parms[1].plot(timplt, py_values[:, 0], marker='o', ls='', color='tab:blue', label='data')
    ax_parms[1].plot(target_times_plt, py_at_targets[:, 0], '-', color='tab:blue', label='fit')
    ax_parms[1].set_ylabel('σᵧ [arcsec/Hz²]')

    ax_parms[2].plot(timplt, px_values[:, 1], marker='o', ls='', color='tab:blue', label='data')
    ax_parms[2].plot(target_times_plt, px_at_targets[:, 1], '-', color='tab:blue', label='fit')
    ax_parms[2].set_ylabel('X shift [arcsec]')

    ax_parms[3].plot(timplt, py_values[:, 1], marker='o', ls='', color='tab:blue', label='data')
    ax_parms[3].plot(target_times_plt, py_at_targets[:, 1], '-', color='tab:blue', label='fit')
    ax_parms[3].set_ylabel('Y shift [arcsec]')

    if tim is not None:
        ax_parms[0].plot(tim.plot_date, params['px0'], color='tab:green', linestyle='', marker='s', label='guess')
        ax_parms[1].plot(tim.plot_date, params['py0'], color='tab:green', linestyle='', marker='s', label='guess')
        ax_parms[2].plot(tim.plot_date, params['px1'], color='tab:green', linestyle='', marker='s', label='guess')
        ax_parms[3].plot(tim.plot_date, params['py1'], color='tab:green', linestyle='', marker='s', label='guess')

    # Calculate the ptp (peak-to-peak) for the first pair of subplots
    ylim0 = ax_parms[0].get_ylim()
    ylim1 = ax_parms[1].get_ylim()
    ptp_01 = max(np.ptp(ylim0), np.ptp(ylim1))

    # Center the y-limits around the mean and set the same ptp
    mean_0 = np.mean(ylim0)
    mean_1 = np.mean(ylim1)
    ax_parms[0].set_ylim(mean_0 - ptp_01*1.5, mean_0 + ptp_01*1.5)
    ax_parms[1].set_ylim(mean_1 - ptp_01*1.5, mean_1 + ptp_01*1.5)

    # Calculate the ptp (peak-to-peak) for the second pair of subplots
    ylim2 = ax_parms[2].get_ylim()
    ylim3 = ax_parms[3].get_ylim()
    ptp_23 = max(np.ptp(ylim2), np.ptp(ylim3), 200)

    # Center the y-limits around the mean and set the same ptp
    mean_2 = np.mean(ylim2)
    mean_3 = np.mean(ylim3)
    ax_parms[2].set_ylim(mean_2 - ptp_23*1.5, mean_2 + ptp_23*1.5)
    ax_parms[3].set_ylim(mean_3 - ptp_23*1.5, mean_3 + ptp_23*1.5)

    ax_parms[0].legend(frameon=True, framealpha=0.5)
    ax_parms[1].legend(frameon=True, framealpha=0.5)
    ax_parms[2].legend(frameon=True, framealpha=0.5)
    ax_parms[3].legend(frameon=True, framealpha=0.5)

    ax = ax_parms[3]

    total_minutes = (target_times_plt[-1] - target_times_plt[0])/24/60
    minticks = 3
    maxticks = 5
    major_locator = AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(major_locator)

    # Create formatter for major ticks
    formatter = AutoDateFormatter(major_locator)
    if total_minutes < 1:
        formatter.scaled[1.0] = '%H:%M:%S.%f'  # Show milliseconds for very short durations
    elif total_minutes < 60:
        formatter.scaled[1.0] = '%H:%M:%S'  # Show seconds for short durations
    else:
        formatter.scaled[1.0] = '%H:%M'  # For longer durations, show only hours and minutes
    formatter.scaled[1 / 24] = '%H:%M'
    formatter.scaled[1 / (24 * 60)] = '%H:%M'
    formatter.scaled[1 / (24 * 60 * 60)] = '%H:%M:%S'
    ax.xaxis.set_major_formatter(formatter)

    # Create locator for minor ticks
    minor_locator = AutoDateLocator(minticks=minticks * 5, maxticks=minticks * 5)  # More minor ticks
    ax.xaxis.set_minor_locator(minor_locator)
    ax.set_xlabel('Time [UT]')

    fig_parms.tight_layout()
    if save_fig:
        figname = os.path.join(workdir, f'fig-refrac_corr_parms_{target_times[0].strftime("%Y%m%d")}.jpg')
        print(f"Saving refrac corr params figure: {figname}")
        fig_parms.savefig(figname, dpi=300, bbox_inches='tight')
    return fig_parms


def interpolate_rfrcorr_file(filename, target_times, methods=('fit:linear', 'fit:linear'), do_plot=False, save_fig=True,
                             workdir='./'):
    if isinstance(filename, str):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filename)
            tims = pd.to_datetime(df['time'])
            px_values = np.vstack([[df['px0'].values, df['px1'].values]]).T
            py_values = np.vstack([[df['py0'].values, df['py1'].values]]).T
        else:
            # Handle single JSON file
            px, py, tim = read_rfrcorr_parms_json(filename)
            tims = [tim]
            px_values = [px]
            py_values = [py]
    elif isinstance(filename, list) and filename[0].endswith('.csv'):
        # Handle list of a CSV file
        df = pd.read_csv(filename[0])
        tims = pd.to_datetime(df['time'])
        px_values = np.vstack([[df['px0'].values, df['px1'].values]]).T
        py_values = np.vstack([[df['py0'].values, df['py1'].values]]).T
    else:
        # Handle list of JSON files
        filename = sorted(filename)
        tims = []
        px_values = []
        py_values = []
        for f in filename:
            px, py, tim = read_rfrcorr_parms_json(f)
            if tim is not None:
                tims.append(tim)
                px_values.append(px)
                py_values.append(py)

    px_values = np.array(px_values)
    py_values = np.array(py_values)

    # Perform interpolation to find px, py at the target times
    if len(tims) > 1:
        px_at_targets, py_at_targets = interpolate_rfrcorr_params(tims, px_values, py_values, target_times,
                                                                  methods)
        if do_plot:
            plot_interpolated_params(tims, px_values, py_values, target_times, px_at_targets, py_at_targets, save_fig=save_fig,
                                     workdir=workdir)
        return px_at_targets, py_at_targets
    else:
        return px_values, py_values


def interpolate_rfrcorr_params(times, px_values, py_values, target_times, methods=('fit:linear', 'fit:linear')):
    """
    Interpolate/fit px and py values at the target times.

    :param times: List of time strings in the format %Y-%m-%dT%H:%M:%S.
    :type times: list
    :param px_values: List of px arrays.
    :type px_values: list
    :param py_values: List of py arrays.
    :type py_values: list
    :param target_times: List or scalar time string(s) in the format %Y-%m-%dT%H:%M:%S for which interpolation/ fitting is required.
    :type target_times: list or str
    :param methods: fit/Interpolation methods ('fit:linear', 'fit:quadratic', 'interp:linear', 'interp:nearest', 'interp:nearest-up', 'interp:zero', 'interp:quadratic', 'interp:cubic', 'interp:previous', 'interp:next'), defaults to ('interp:linear', 'interp:linear').
    :type methods: tuple, optional
    :return: List of interpolated/fit px and py arrays for each target time if target_times is a list, or a single interpolated/fit px and py array if target_times is a scalar.
    :rtype: list
    """

    times = [t if isinstance(t, datetime) else datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in times]
    times_numeric = Time(times).mjd

    if isinstance(target_times, str) or isinstance(target_times, datetime):
        target_times = [target_times]

    target_times_numeric = Time([
        t if isinstance(t, datetime) else datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in target_times
    ]).mjd

    px_values = np.array(px_values)
    py_values = np.array(py_values)
    # print(f'shape of px_values: {px_values.shape}')

    if isinstance(methods, str):
        methods = (methods, methods)
    elif isinstance(methods, (list, tuple)) and len(methods) == 2:
        methods = methods
    else:
        raise ValueError("methods should be a string or a list/tuple of two strings.")

    def apply_method(values, method):
        result = []
        if method.startswith('fit'):
            if method == 'fit:linear':
                deg = 1
            elif method == 'fit:quadratic':
                deg = 2
            # Perform a 2nd-degree polynomial fit
            for col in range(values.shape[1]):
                coefs = np.polyfit(times_numeric, values[:, col], deg)
                fit_func = np.poly1d(coefs)
                result.append(fit_func(target_times_numeric))
        elif method.startswith('interp:'):
            interp_kind = method.split(':')[1]
            # Perform interpolation using scipy's interp1d
            for col in range(values.shape[1]):
                interp_func = interp1d(times_numeric, values[:, col], kind=interp_kind, fill_value="extrapolate")
                result.append(interp_func(target_times_numeric))
        else:
            raise ValueError(
                f"Unknown method: {method}. Supported methods are 'fit:linear', 'fit:quadratic', and 'interp:<method>'.")

        return np.array(result).T

    p1_values = np.vstack([px_values[:, 0], py_values[:, 0]]).T
    p2_values = np.vstack([px_values[:, 1], py_values[:, 1]]).T
    p1_at_targets = apply_method(p1_values, methods[0])
    p2_at_targets = apply_method(p2_values, methods[1])

    px_at_targets = np.vstack([p1_at_targets[:, 0], p2_at_targets[:, 0]]).T
    py_at_targets = np.vstack([p1_at_targets[:, 1], p2_at_targets[:, 1]]).T

    return px_at_targets, py_at_targets


def correct_images(image_series, px, py, freqs):
    """
    Correct image shifts based on refraction parameters.

    :param image_series: np.ndarray
        Image series to be corrected.
    :param px: float
        Refraction correction coefficient in x direction.
    :param py: float
        Refraction correction coefficient in y direction.
    :param freqs: list
        List of frequencies in MHz.
    :return: np.ndarray
        Corrected images.
    """
    nf, ny, nx = image_series.shape
    corrected_images = np.zeros((nf, ny, nx))
    shift_x, shift_y = pxy2shifts(px, py, freqs)
    for idx in range(nf):
        corrected_images[idx] = np.roll(image_series[idx], shift=(int(-shift_x[idx]), int(-shift_y[idx])), axis=(1, 0))
    return corrected_images


def norm_to_percent(data, minpercent=None, maxpercent=None, logscale=False):
    """
    Normalize the input array based on a specified percentile.

    This function scales the input data so that the specified percentile value becomes 0
    and the maximum value becomes 1. Values below the percentile are set to 0.

    :param data: A numpy array containing the data to be normalized.
    :type data: np.ndarray
    :param minpercent: The percentile used to determine the minimum value for scaling.
                        If None, the minimum is set to 0 percentile.
    :type minpercent: float, optional
    :return: The normalized data array.
    :rtype: np.ndarray
    """
    # Compute the maximum value ignoring NaNs

    # Set default percentile if none provided
    if minpercent is None:
        minpercent = 0
    if maxpercent is None:
        maxpercent = 100

    # Compute the minimum value based on the given percentile, ignoring NaNs
    dmin = np.nanmax(data) * minpercent / 100
    dmax = np.nanmax(data) * maxpercent / 100

    # Normalize the data
    if logscale:
        dmin = max(dmin, 1)
        newdata = (np.log(data) - np.log(dmin)) / (np.log(dmax) - np.log(dmin))
    else:
        newdata = (data - dmin) / (dmax - dmin)
    newdata[newdata < 0] = 0
    newdata[newdata > 1] = 1

    return newdata


def norm_to_value(data, minval=None, maxval=None, logscale=False):
    """
    Normalize the input array based on a specified value.

    This function scales the input data so that the specified value becomes 0
    and the maximum value becomes 1. Values below the value are set to 0.

    :param data: A numpy array containing the data to be normalized.
    :type data: np.ndarray
    :param minval: The value used to determine the minimum value for scaling.
                        If None, the minimum is set to 0 value.
    :type minval: float, optional
    :return: The normalized data array.
    :rtype: np.ndarray
    """
    # Compute the maximum value ignoring NaNs

    # Set default value if none provided
    if minval is None:
        minval = np.nanmax(data)
    if maxval is None:
        maxval = np.nanmax(data)

    # Compute the minimum value based on the given value, ignoring NaNs
    dmin = minval
    dmax = maxval

    # Normalize the data
    if logscale:
        dmin = max(dmin, 1)
        newdata = (np.log(data) - np.log(dmin)) / (np.log(dmax) - np.log(dmin))
    else:
        newdata = (data - dmin) / (dmax - dmin)
    newdata[newdata < 0] = 0
    newdata[newdata > 1] = 1

    return newdata


def data1d_to_rgba(data, color, alpha=1.0, minpercent=0):
    data_normalized = norm_to_percent(data.ravel(order='F'), minpercent=minpercent)
    npix = data.size
    rgba_array = np.zeros((npix, 4), dtype=np.uint8)
    rgba_array[..., :3] = (color[:3] * 255).astype(np.uint8)
    rgba_array[..., 3] = (data_normalized * 255).astype(np.uint8) * alpha
    return rgba_array


# def define_filename(rmap, prefix="", ext=".json", delimiter="_"):
#     """
#     Define the filename for saving refraction correction parameters.
#
#     :return: str
#         Defined filename.
#     """
#     filename_ = []
#     if hasattr(rmap, 'observatory'):
#         if rmap.observatory != "":
#             filename_.append(f"{rmap.observatory.rstrip('-fast')}")
#     if hasattr(rmap, 'detector'):
#         if rmap.detector != "":
#             filename_.append(f"{rmap.detector}")
#     if hasattr(rmap, 'date'):
#         filename_.append(rmap.date.to_datetime().strftime("%Y-%m-%dT%H%M%S"))
#     filename = delimiter.join(filename_)
#     return f"{prefix}{filename}{ext}"


def define_filename(rmap, prefix="", ext=".json", delimiter="_", directory=".", get_latest_version=False):
    """
    Define the filename for saving refraction correction parameters with the highest version.

    :return: str
        Defined filename with the highest version tag if it exists.
    """

    def get_latest_versioned_filename(base_filename, ext, directory='.'):
        """
        Find the latest versioned filename with a given base name and extension in a specified directory.

        Parameters:
        base_filename (str): The base filename without the version tag and extension.
        ext (str): The file extension (e.g., ".json").
        directory (str): The directory to search for files.

        Returns:
        str: The latest versioned filename or the base filename if no versioned files are found.
        """
        pattern = re.compile(rf"{re.escape(base_filename)}(\.v\d+)?{re.escape(ext)}")
        files = glob(os.path.join(directory, f"{base_filename}*{ext}"))
        versioned_files = [f for f in files if pattern.match(os.path.basename(f))]

        if not versioned_files:
            return os.path.join(directory, base_filename + ext)

        version_numbers = [re.search(r'\.v(\d+)', f) for f in versioned_files]
        version_numbers = [int(m.group(1)) for m in version_numbers if m]

        if version_numbers:
            highest_version = max(version_numbers)
            latest_versioned_filename = f"{base_filename}.v{highest_version}{ext}"
        else:
            latest_versioned_filename = base_filename + ext

        return os.path.join(directory, latest_versioned_filename)

    filename_ = []
    if hasattr(rmap, 'observatory') and rmap.observatory != "":
        filename_.append(f"{rmap.observatory.rstrip('-fast')}")
    if hasattr(rmap, 'detector') and rmap.detector != "":
        filename_.append(f"{rmap.detector}")
    if hasattr(rmap, 'date'):
        if ext == ".json":
            filename_.append(rmap.date.strftime("%Y-%m-%dT%H%M%S"))
        elif ext == ".csv":
            filename_.append(rmap.date.strftime("%Y-%m-%d"))
        else:
            filename_.append(rmap.date.strftime("%Y-%m-%dT%H%M%S"))
    base_filename = delimiter.join(filename_)
    base_filename = f"{prefix}{base_filename}"
    #
    # import pdb;
    # pdb.set_trace()

    if get_latest_version:
        return get_latest_versioned_filename(base_filename, ext, directory)
    else:
        return os.path.join(directory, base_filename + ext)


def define_timestring(rmap, prefix="", delimiter=" "):
    """
    Define the time string for saving refraction correction parameters.

    :return: str
        Defined time string.
    """
    filename_ = []
    if hasattr(rmap, 'observatory'):
        if rmap.observatory != "":
            filename_.append(f"{rmap.observatory.rstrip('-fast')}")
    if hasattr(rmap, 'detector'):
        if rmap.detector != "":
            filename_.append(f"{rmap.detector}")
    if len(filename_) == 2:
        filename_ = ["/".join(filename_)]
    if hasattr(rmap, 'wavelength'):
        if rmap.wavelength != "" and rmap.wavelength is not None:
            wavelength = rmap.wavelength.value
            waveunit = r'$\AA$' if rmap.waveunit == 'angstrom' else rmap.waveunit.to_string()
            filename_.append(f"{wavelength:.1f} {waveunit}")
    if hasattr(rmap, 'date'):
        filename_.append(rmap.date.to_datetime().strftime("%Y-%m-%d %H:%M:%S UT"))
    filename = delimiter.join(filename_)
    return f"{prefix}{filename}"
