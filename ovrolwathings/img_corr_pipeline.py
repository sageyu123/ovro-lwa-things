import argparse
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import hvpy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sunpy
from astropy.coordinates import SkyCoord
from astropy.time import Time
from hvpy.datasource import DataSource
from sunpy.coordinates import Helioprojective
from sunpy.map import Map
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.util.config import get_and_create_download_dir
from tqdm import tqdm

import ovrolwathings
from ovrolwasolar.utils import recover_fits_from_h5
# from PyQt5.QtWidgets import QApplication
from ovrolwathings import refr_corr_tool as rct
from ovrolwathings.download_utils import download_ovrolwa
from ovrolwathings.utils import apply_intensity_threshold, cal_pix_shifts, cal_pxy, define_filename, define_timestring, \
    interpolate_rfrcorr_file, norm_to_percent, pxy2shifts
from suncasa.io import ndfits

# use pwd as base directory
base_dir = os.getcwd()
 #Path(ovrolwathings.__file__).parent

# Dictionary mapping suvi_passband to DataSource values
suvi_passband_map = {
    '94': DataSource.SUVI_94.value,
    '131': DataSource.SUVI_131.value,
    '171': DataSource.SUVI_171.value,
    '195': DataSource.SUVI_195.value,
    '284': DataSource.SUVI_284.value,
    '304': DataSource.SUVI_304.value
}


def update_alpha(im, threshold=0.0, width=0.1, alpha_min=0, alpha_max=1):
    """
    Update the alpha channel of an image with a smooth transition based on intensity values.

    This function modifies the alpha channel of an image to create a smooth transition
    around a specified intensity threshold. The transition is controlled by a width parameter,
    allowing for fine-tuned adjustments to the alpha transparency effect.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The image object to modify.
    threshold : float
        The intensity threshold around which the alpha transition occurs.
    width : float, optional
        The width of the transition zone around the threshold as a fraction of the threshold value.
        Default is 0.1.
    alpha_min : float, optional
        The minimum alpha value for intensities below the threshold. Default is 0.
    alpha_max : float, optional
        The maximum alpha value for intensities above the threshold. Default is 1.

    Returns
    -------
    None
        This function modifies the image in place and does not return any value.

    Example
    -------
    >>> im = lwa_reprojected.plot(axes=ax, autoalign=True, alpha=1.0, cmap='afmhot')
    >>> update_alpha(im, threshold=50e6, width=0.05, alpha_min=0, alpha_max=1)
    """
    intensity = im.get_array()
    rgba = im.to_rgba(intensity)  # Get the RGBA array

    # Create a smooth transition for the alpha channel
    # alpha = 1 / (1 + np.exp(-(intensity - threshold) / (width*threshold)))
    alpha = alpha_min + (alpha_max - alpha_min) / (1 + np.exp(-(intensity - threshold) / (width * threshold)))

    rgba[..., -1] = alpha  # Update the alpha channel in the RGBA array
    im.set_array(rgba)  # Update the image with the new RGBA array


def enhance_offdisk_corona(smap):
    hpc_coords = all_coordinates_from_map(smap)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / smap.rsun_obs
    rsun_step_size = 0.01
    rsun_array = np.arange(1, r.max(), rsun_step_size)
    y = np.array([smap.data[(r > this_r) * (r < this_r + rsun_step_size)].mean()
                  for this_r in rsun_array])
    params = np.polyfit(rsun_array[rsun_array < 1.5],
                        np.log(y[rsun_array < 1.5]), 1)
    # best_fit = np.exp(np.poly1d(params)(rsun_array))
    scale_factor = np.exp((r - 1) * (-params[0]) / 2)
    scale_factor[r < 1] = 1
    scaled_smap = sunpy.map.Map(smap.data * scale_factor, smap.meta)
    return scaled_smap


# def run_imgcorr_app(lwafile, background_map):
#     # app = QApplication([])  # Add this line
#     imgcorr = rct.ImageCorrectionApp(lwafile=lwafile, background_map=background_map)
#     imgcorr.show()
#     # sys.exit(app.exec_())

# def get_submap_mean(smap, x1, y1, x2, y2):
#     submap = smap.submap([x1 * u.arcsec, x2 * u.arcsec], [y1 * u.arcsec, y2 * u.arcsec])
#     return np.nanmean(submap.data)


def main(mode, timestamps, freqplts, mnorm=None, level='lev1', filetype='hdf', specmode='mfs',
         auto_corr=False,
         refrac_corr=False,
         workdir='.',
         datadir='.',
         scaled_suvi=True,
         suvi_passband=171,
         show_ax_grid=True, alpha=0.7, minpercent=5, draw_contours=False,
         fov=[16000, 16000],
         fov_center=[0, 0],
         roi=None,
         roi_percentile=None,
         dual_panel=False,
         get_latest_version=False,
         timediff_tol=None, sshconfig='ssh-to-data.private.config',
         snr_threshold=0.0,
         rfrcor_parm_files=[], interp_method=('fit:linear', 'fit:linear'), trajectory_file='', overwrite=False):
    do_plot = False if refrac_corr else True
    try:
        if os.path.isabs(workdir):
            if not os.path.exists(workdir):
                os.makedirs(workdir, exist_ok=True)
            os.chdir(workdir)
    except:
        pass

    downloaded_files = download_ovrolwa(timestamps=timestamps,
                                        data_dir=datadir,
                                        mode=mode, level=level, filetype=filetype,
                                        timediff_tol=timediff_tol,
                                        specmode=specmode,
                                        config=os.path.join(base_dir, sshconfig))
    # print("Downloaded files:", downloaded_files)
    ntim = len(timestamps)
    if roi is None:
        specs = None
    else:
        specs = {i: [[]] * ntim for i in range(len(roi))}
    if do_plot:
        plt.ioff()
        print(rfrcor_parm_files)
        if rfrcor_parm_files:
            px_at_targets, py_at_targets = interpolate_rfrcorr_file(rfrcor_parm_files, timestamps, interp_method,
                                                                    do_plot=True, workdir=workdir)
    else:
        px_at_targets = []
        py_at_targets = []

    # for tidx, (downloaded_file, timestamp) in enumerate(tqdm(zip(downloaded_files, timestamps))):
    #     if downloaded_file is None:
    #         print(f'file: {downloaded_file}, timestamp: {timestamp}')
    #     else:
    #         print(f'file: {os.path.basename(downloaded_file)}, timestamp: {timestamp}')

    suvi_key = f'suvi_{suvi_passband}'
    for tidx, (downloaded_file, timestamp) in enumerate(tqdm(zip(downloaded_files, timestamps))):
        print(f'file: {downloaded_file}, timestamp: {timestamp}')
        # if tidx > 0: continue
        if downloaded_file is None:
            if roi is not None:
                for ridx in specs:
                    specs[ridx][tidx] = None
            continue
        timestr = timestamp.strftime("%Y%m%dT%H%M%S")

        if do_plot:
            if filetype == 'hdf':
                ## Reading data from fch hdf files is slow, therefore only read meta first
                meta = recover_fits_from_h5(downloaded_file, return_data=True, return_meta_only=True)
            else:
                meta, data = ndfits.read(downloaded_file)
                if 'cfreqs' not in meta:
                    meta['cfreqs'] = meta['ref_cfreqs']

            figname = define_filename(meta['header'], prefix="fig-", ext=".jpg").replace(' ', '_')
            if 'figpath' not in locals():
                figpath = figname.split('T')[0]
                if not os.path.exists(figpath):
                    os.makedirs(figpath, exist_ok=True)
            figname = os.path.join(figpath, figname)
            if not overwrite and os.path.exists(figname):
                if roi is None:
                    print(f"Skipping existing file: {figname}")
                    continue

            # print(f'{downloaded_file} n of freqs: {ncfreqs}')

            if filetype == 'hdf':
                meta, data = recover_fits_from_h5(downloaded_file, return_data=True)
            header = meta['header']
            delta_x = header['CDELT1']
            delta_y = header['CDELT2']
            cfreq_unit = u.Unit(meta['header']['CUNIT3'])
            ncfreqs = len(meta['cfreqs'])
            corrected_data = np.squeeze(data)
            if not np.all(np.diff(meta['cfreqs']) > 0):
                cfreqs_sort_idx = np.argsort(meta['cfreqs'])
                meta['cfreqs'] = meta['cfreqs'][cfreqs_sort_idx]
                cfreqshz = meta['cfreqs']
                cfreqsmhz = cfreqshz / 1e6
                corrected_data = corrected_data[cfreqs_sort_idx, ...]
            # print(f'cfreqs [Hz]: {cfreqshz}')
            lwamap = Map(corrected_data[0], meta['header'])
            if rfrcor_parm_files:
                if len(px_at_targets) > 0 and len(py_at_targets) > 0:
                    px, py = px_at_targets[tidx], py_at_targets[tidx]
                    print(f'using px, py from interpolation: {px}, {py}')
            else:
                # params_filename = define_filename(lwamap, prefix="refrac_corr_", ext=".json",
                #                                   get_latest_version=get_latest_version)
                # params_filepath = os.path.join(workdir, params_filename)
                # px, py, tim = read_rfrcorr_parms_json(params_filepath)
                px = [0.0, 0.0]
                py = [0.0, 0.0]

            if auto_corr:
                img2proc = [apply_intensity_threshold(img, per_thrshd=95) for img in corrected_data]
                ref_image = img2proc[-1]
                pix_shifts = cal_pix_shifts(img2proc, ref_image)
                ## py_ comes first before px_ because the processing is in the image plane, whose x y axis are the y x axis of the 2D array.
                py_, px_ = cal_pxy(cfreqsmhz, pix_shifts)
                if2 = cfreqsmhz ** (-2)
                shift_x_pix = np.polyval(px_, if2)
                ## The minus sign is used because the direction of y of the image and the array are opposite.
                shift_y_pix = -np.polyval(py_, if2)
                shifts_x, shifts_y = shift_x_pix * delta_x + px[1], shift_y_pix * delta_y + py[1]
                shifts_x_orig, shifts_y_orig = pxy2shifts(px, py, freqplts)
                print(f'original shifts in pixels: {shifts_x_orig / delta_x}, {shifts_y_orig / delta_y}')
                print(f'auto corr shift in pixels: {shift_x_pix}, {shift_y_pix}')
            else:
                shifts_x, shifts_y = pxy2shifts(px, py, freqplts)

            if roi is not None:
                shifts_x_all, shifts_y_all = pxy2shifts(px, py, cfreqsmhz)
                # print(f'shifts_x_all: {shifts_x_all}, shifts_y_all: {shifts_y_all}')
                roi_coords = {}
                for ridx, roi_ in enumerate(roi):
                    specs[ridx][tidx] = np.full((ncfreqs), np.nan)
                # print(f'cfreqsmhz: {cfreqsmhz}')
                for idx, freq in enumerate(cfreqsmhz):
                    lwamap = Map(corrected_data[idx, ...], meta['header'])
                    lwamap = lwamap.shift_reference_coord(-shifts_x_all[idx] * u.arcsec,
                                                          -shifts_y_all[idx] * u.arcsec)
                    # print(lwamap.data.shape)
                    for ridx, roi_ in enumerate(roi):
                        if idx == 0:
                            x1, y1, x2, y2 = roi_
                            top_right_roi = SkyCoord(x2 * u.arcsec, y2 * u.arcsec, frame=lwamap.coordinate_frame)
                            bottom_left_roi = SkyCoord(x1 * u.arcsec, y1 * u.arcsec, frame=lwamap.coordinate_frame)
                            roi_coords[ridx] = (bottom_left_roi, top_right_roi)
                        bottom_left_roi, top_right_roi = roi_coords[ridx]
                        lwamap_sub = lwamap.submap(bottom_left_roi, top_right=top_right_roi)
                        # print(lwamap_sub.data)
                        # import IPython;
                        # IPython.embed()
                        if roi_percentile is None:
                            specs[ridx][tidx][idx] = np.nanmean(lwamap_sub.data)
                        else:
                            specs[ridx][tidx][idx] = np.nanpercentile(lwamap_sub.data, roi_percentile)
                        # print(f'ridx: {ridx}, idx: {idx}, freq: {freq}, mean: {specs[ridx][tidx][idx]}')
                for ridx, spec in specs.items():
                    specs[ridx][tidx] = np.array(specs[ridx][tidx])[cfreqs_sort_idx]
                continue

            lasco_c3_jp2_file = get_and_create_download_dir() + f"/LASCO_C3_{timestr}.jp2"
            if not os.path.exists(lasco_c3_jp2_file):
                lasco_c3_jp2_file = hvpy.save_file(hvpy.getJP2Image(timestamp,
                                                                    DataSource.LASCO_C3.value),
                                                   filename=lasco_c3_jp2_file, overwrite=True)
            lasco_c2_jp2_file = get_and_create_download_dir() + f"/LASCO_C2_{timestr}.jp2"
            if not os.path.exists(lasco_c2_jp2_file):
                lasco_c2_jp2_file = hvpy.save_file(hvpy.getJP2Image(timestamp,
                                                                    DataSource.LASCO_C2.value),
                                                   filename=lasco_c2_jp2_file, overwrite=True)
            suvi_jp2_file = get_and_create_download_dir() + f"/SUVI_{suvi_passband}_{timestr}.jp2"
            if not os.path.exists(suvi_jp2_file):
                suvi_data_source = suvi_passband_map.get(suvi_passband, DataSource.SUVI_171.value)
                suvi_jp2_file = hvpy.save_file(hvpy.getJP2Image(timestamp,
                                                                suvi_data_source),
                                               filename=suvi_jp2_file, overwrite=True)
            lasco_c3_map = Map(lasco_c3_jp2_file)
            lasco_c2_map = Map(lasco_c2_jp2_file)
            suvi_map = Map(suvi_jp2_file)

            if scaled_suvi:
                scaled_suvi_map = enhance_offdisk_corona(suvi_map)
            else:
                scaled_suvi_map = suvi_map

            bkgmaps_orig = {'lasco_c3': lasco_c3_map, 'lasco_c2': lasco_c2_map, suvi_key: scaled_suvi_map}

            projected_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                                       obstime=lwamap.observer_coordinate.obstime,
                                       frame='helioprojective',
                                       observer=lwamap.observer_coordinate,
                                       rsun=suvi_map.coordinate_frame.rsun)
            bkgmaps = {}
            for k, bkgmap in bkgmaps_orig.items():
                projected_header = sunpy.map.make_fitswcs_header(bkgmap.data.shape,
                                                                 projected_coord,
                                                                 scale=u.Quantity(bkgmap.scale),
                                                                 instrument=bkgmap.instrument,
                                                                 wavelength=bkgmap.wavelength)

                with Helioprojective.assume_spherical_screen(bkgmap.observer_coordinate):
                    bkgmaps[k] = bkgmap.reproject_to(projected_header)

            # top_right = SkyCoord(15000 * u.arcsec, 15000 * u.arcsec, frame=bkgmaps['lasco_c3'].coordinate_frame)
            # bottom_left = SkyCoord(-15000 * u.arcsec, -15000 * u.arcsec, frame=bkgmaps['lasco_c3'].coordinate_frame)
            top_right = SkyCoord((fov_center[0] + fov[0] / 2) * u.arcsec, (fov_center[1] + fov[1] / 2) * u.arcsec,
                                 frame=bkgmaps['lasco_c3'].coordinate_frame)
            bottom_left = SkyCoord((fov_center[0] - fov[0] / 2) * u.arcsec, (fov_center[1] - fov[1] / 2) * u.arcsec,
                                   frame=bkgmaps['lasco_c3'].coordinate_frame)
            top_right_debuff = SkyCoord((fov_center[0] + 0.98 * fov[0] / 2) * u.arcsec,
                                        (fov_center[1] + 0.98 * fov[1] / 2) * u.arcsec,
                                        frame=bkgmaps['lasco_c3'].coordinate_frame)
            bottom_left_debuff = SkyCoord((fov_center[0] - 0.98 * fov[0] / 2) * u.arcsec,
                                          (fov_center[1] - 0.98 * fov[1] / 2) * u.arcsec,
                                          frame=bkgmaps['lasco_c3'].coordinate_frame)
            bkgmaps['lasco_c3'] = bkgmaps['lasco_c3'].submap(bottom_left,
                                                             top_right=top_right)
            try:
                bkgmaps['lasco_c2'] = bkgmaps['lasco_c2'].submap(bottom_left,
                                                                 top_right=top_right)
                bkgmaps[suvi_key] = bkgmaps[suvi_key].submap(bottom_left,
                                                             top_right=top_right)
            except:
                pass

            axs = []
            fig = plt.figure(figsize=(12, 6) if dual_panel else (6, 6), constrained_layout=True)
            if dual_panel:
                axs.append(fig.add_subplot(121, projection=bkgmaps['lasco_c3']))
                axs.append(fig.add_subplot(122, projection=bkgmaps['lasco_c3']))
            else:
                axs.append(fig.add_subplot(projection=bkgmaps['lasco_c3']))

            for ax in axs:
                bkgmaps['lasco_c3'].plot(axes=ax, cmap='gray', zorder=-3)
                bkgmaps['lasco_c2'].plot(axes=ax, cmap='gray', zorder=-2, autoalign=True)
                bkgmaps[suvi_key].plot(axes=ax, clip_interval=(1, 99.9) * u.percent, autoalign=True, cmap='gray',
                                       zorder=-1)

            cmap = plt.get_cmap('jet')
            colors_frac = np.linspace(0, 1, len(freqplts))
            colors = cmap(colors_frac)

            for idx, freqplt in enumerate(freqplts):
                fidx = np.nanargmin(np.abs(cfreqshz * cfreq_unit - freqplt))

                if snr_threshold > 0:
                    snr = np.nanmax(corrected_data[fidx, ...]) / np.nanstd(corrected_data[fidx, :10, :])
                    print(f'freq: {freqplt}, snr: {snr}')
                    if snr < snr_threshold:
                        continue
                lwamap = Map(corrected_data[fidx, ...], meta['header'])
                projected_header_lwa = sunpy.map.make_fitswcs_header(lwamap.data.shape,
                                                                     projected_coord,
                                                                     scale=u.Quantity(lwamap.scale),
                                                                     instrument='ovrolwa',
                                                                     wavelength=cfreqshz[fidx] * u.Hz)

                with Helioprojective.assume_spherical_screen(lwamap.observer_coordinate):
                    lwa_reprojected = lwamap.reproject_to(projected_header_lwa)

                # print(f'shift in freq {freqplt}: {shifts_x[idx]}, {shifts_y[idx]}')
                lwa_reprojected = lwa_reprojected.shift_reference_coord(-shifts_x[idx] * u.arcsec,
                                                                        -shifts_y[idx] * u.arcsec)

                lwa_reprojected_crop = lwa_reprojected.submap(bottom_left_debuff,
                                                              top_right=top_right_debuff)
                # lwa_reprojected_crop = lwa_reprojected
                # lwadatamax = np.nanmax(lwa_reprojected.data)

                # im = lwa_reprojected.plot(axes=ax, norm=mcolors.Normalize(vmax=lwadatamax, vmin=5e6), autoalign=True, alpha=1.0,
                #                           cmap='afmhot')

                ax = axs[0]
                im = lwa_reprojected_crop.plot(axes=ax, autoalign=True, norm=mnorm, alpha=1.0,
                                               cmap='gray')

                intensity = im.get_array()
                rgba = im.to_rgba(intensity)  # Get the RGBA array

                rgba[..., 0] = colors[idx][0]  # Red
                rgba[..., 1] = colors[idx][1]  # Green
                rgba[..., 2] = colors[idx][2]  # Blue

                opacity_array = norm_to_percent(lwa_reprojected_crop.data,
                                                minpercent=minpercent,
                                                logscale=True)

                rgba[..., 3] = opacity_array * alpha  # Update the alpha channel in the RGBA array
                im.set_array(rgba)  # Update the image with the new RGBA array
                if draw_contours:
                    lwa_reprojected_crop.draw_contours(axes=ax, colors=[colors[idx]], levels=[minpercent] * u.percent,
                                                       alpha=alpha, linewidths=0.35)

                if dual_panel:
                    ax = axs[1]
                    lwa_reprojected_crop.draw_contours(axes=ax, colors=[colors[idx]], levels=[minpercent] * u.percent,
                                                       alpha=0.5, linewidths=0.35)

            ax = axs[0]
            ax.text(0.02, 0.98, define_timestring(lwamap, delimiter=' '), transform=ax.transAxes,
                    color='white',
                    va='top', ha='left')
            ax.text(0.02, 0.10, define_timestring(bkgmaps_orig['lasco_c3'], delimiter=' '), transform=ax.transAxes,
                    color='white',
                    va='bottom', ha='left')
            ax.text(0.02, 0.06, define_timestring(bkgmaps_orig['lasco_c2'], delimiter=' '), transform=ax.transAxes,
                    color='white',
                    va='bottom', ha='left')
            ax.text(0.02, 0.02, define_timestring(bkgmaps_orig[suvi_key], delimiter=' '), transform=ax.transAxes,
                    color='white',
                    va='bottom', ha='left')

            fig.tight_layout()
            # Add colorbar
            freqnorm = plt.Normalize(vmin=freqplts.value.min(), vmax=freqplts.value.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=freqnorm)
            sm.set_array([])
            ax = axs[-1]
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
            cbar.set_label('Frequency [MHz]')

            for ax in axs:
                ax.set_title("")
                ax.set_xlabel('Solar-X [arcsec]')
                ax.set_ylabel('Solar-Y [arcsec]')
                ax.grid(show_ax_grid)

            # fig.tight_layout()
            fig.savefig(figname, dpi=200, bbox_inches='tight')
            print(f"Saved figure: {figname}")
            plt.close(fig)
        plt.ion()

        if refrac_corr:
            # imgcorr = rct.ImageCorrectionApp(lwafile=downloaded_file, background_map= [lasco_c3_jp2_file,lasco_c2_jp2_file, suvi_jp2_file])
            # imgcorr.show()
            print(f"t{tidx + 1}/{ntim} --------------- processing refrac corr for  timestamp: {timestamp}")
            # run_imgcorr_app(lwafile=downloaded_file, background_map = [lasco_c3_jp2_file,lasco_c2_jp2_file, suvi_jp2_file])
            # Convert the background map files from Path objects to strings if necessary
            lasco_c3_jp2_file = str(lasco_c3_jp2_file)
            lasco_c2_jp2_file = str(lasco_c2_jp2_file)
            suvi_jp2_file = str(suvi_jp2_file)

            # Construct the command to run
            command = [
                'python', rct.__file__,
                '--lwafile', downloaded_file,
                '--background_map', lasco_c3_jp2_file, lasco_c2_jp2_file, suvi_jp2_file,
                '--freqs', ','.join(map(str, freqplts.value)),
                '--trajectory_file', trajectory_file,
            ]
            print(' '.join(command))
            # Execute the command
            subprocess.run(command)
    if roi is not None:
        time_axis = Time(timestamps)
        freq_axis = cfreqshz
        from suncasa.dspec import dspec as ds
        d = ds.Dspec()
        d.time_axis = time_axis
        d.freq_axis = freq_axis
        timestr = timestamps[0].strftime("%Y%m%dT%H%M%S")

        # import pickle
        # with open(os.path.join(figpath, f'OVROLWA.spec.ROIs.{timestr}.pkl'), 'wb') as f:
        #     pickle.dump([specs, freq_axis, time_axis], f)
        #
        # file_pkl = os.path.join(figpath, f'OVROLWA.spec.ROIs.{timestr}.pkl')
        # with open(file_pkl, 'rb') as f:
        #     specs, freq_axis, time_axis = pickle.load(f)
        for ridx, spec in specs.items():
            max_length = max(len(s) if s is not None else 0 for s in spec)
            print(f"ROI {ridx}: max_length: {max_length}")
            spec = [s if s is not None and len(s) == max_length else np.full(max_length, np.nan) for s in spec]
            spec = np.array(spec)

            if spec.ndim != 2 or spec.shape[1] != len(freq_axis):
                raise ValueError("The array dimensions do not match the frequency axis length.")

            specs[ridx] = spec.T  # Transpose to match expected dimensions for dspec
            d.data = specs[ridx]

            filename = os.path.join(figpath, f'OVROLWA.spec.ROI{ridx}.{timestr}.fits')
            d.tofits(fitsfile=filename, spec_unit='K', observer='OVSAs team', telescope='OVRO-LWA',
                     observatory='Owens Valley Radio Observatory')
            print(f"Saved ROI spec file: {filename}")

    print("Done!")
    # print(f"Processed files: {downloaded_files}")


if __name__ == "__main__":
    '''
    Example usage:
        python img_corr_pipeline.py --mode auto --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --specmode mfs --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 5 --dual_panel --get_latest_version --docorr
        python img_corr_pipeline.py --mode auto --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --filetype fits --specmode mfs --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 5 --dual_panel --get_latest_version --docorr
        python img_corr_pipeline.py --mode auto --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --specmode mfs --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 5 --dual_panel --get_latest_version --rfrcor_parm_files refrac_corr_OVRO-LWA_2024-05-17T205846.json refrac_corr_OVRO-LWA_2024-05-18T003226.v2.json --interp_method interp:linear --timediff_tol 30
        python img_corr_pipeline.py --mode auto --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --filetype hdf --specmode mfs --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 5 --fov 16000 16000 --dual_panel --get_latest_version --rfrcor_parm_files "refrac_corr_OVRO-LWA_2024-05-25T170200.json" "refrac_corr_OVRO-LWA_2024-05-25T183607.json" "refrac_corr_OVRO-LWA_2024-05-25T192406.json" "refrac_corr_OVRO-LWA_2024-05-25T195302.json" "refrac_corr_OVRO-LWA_2024-05-25T202407.json" "refrac_corr_OVRO-LWA_2024-05-25T204802.json" "refrac_corr_OVRO-LWA_2024-05-25T211407.json"  --interp_method interp:cubic fit:linear --timediff_tol 30  --overwrite
        python img_corr_pipeline.py --mode fast --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --filetype hdf --specmode fch --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 25 --fov 16000 16000 --dual_panel --get_latest_version --rfrcor_parm_files "refrac_corr_OVRO-LWA_2024-05-25T170200.json" "refrac_corr_OVRO-LWA_2024-05-25T183607.json" "refrac_corr_OVRO-LWA_2024-05-25T192406.json" "refrac_corr_OVRO-LWA_2024-05-25T195302.json" "refrac_corr_OVRO-LWA_2024-05-25T202407.json" "refrac_corr_OVRO-LWA_2024-05-25T204802.json" "refrac_corr_OVRO-LWA_2024-05-25T211407.json"  --interp_method interp:cubic fit:linear --timediff_tol 8  --overwrite
        python img_corr_pipeline.py --mode auto --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --filetype fits  --specmode mfs --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 5 --fov 24000 24000 --dual_panel --get_latest_version --rfrcor_parm_files  "refrac_corr_OVRO MMA_2024-02-28T164502.json" "refrac_corr_OVRO MMA_2024-02-28T165003.json" "refrac_corr_OVRO MMA_2024-02-28T165604.json" "refrac_corr_OVRO MMA_2024-02-28T170035.json" "refrac_corr_OVRO MMA_2024-02-28T170556.json" "refrac_corr_OVRO MMA_2024-02-28T171207.json" "refrac_corr_OVRO MMA_2024-02-28T171418.json" "refrac_corr_OVRO MMA_2024-02-28T172009.json" "refrac_corr_OVRO MMA_2024-02-28T172800.json" "refrac_corr_OVRO MMA_2024-02-28T173201.json" "refrac_corr_OVRO MMA_2024-02-28T173602.json" "refrac_corr_OVRO MMA_2024-02-28T174604.json" "refrac_corr_OVRO MMA_2024-02-28T175706.json" "refrac_corr_OVRO MMA_2024-02-28T180607.json" "refrac_corr_OVRO MMA_2024-02-28T181209.json" "refrac_corr_OVRO MMA_2024-02-28T181800.json" "refrac_corr_OVRO MMA_2024-02-28T182000.json" "refrac_corr_OVRO MMA_2024-02-28T182501.json" "refrac_corr_OVRO MMA_2024-02-28T182701.json" "refrac_corr_OVRO MMA_2024-02-28T183102.json" "refrac_corr_OVRO MMA_2024-02-28T183302.json" "refrac_corr_OVRO MMA_2024-02-28T184004.json" "refrac_corr_OVRO MMA_2024-02-28T184805.json" "refrac_corr_OVRO MMA_2024-02-28T185707.json" "refrac_corr_OVRO MMA_2024-02-28T190308.json" "refrac_corr_OVRO MMA_2024-02-28T191009.json" "refrac_corr_OVRO MMA_2024-02-28T191200.json" "refrac_corr_OVRO MMA_2024-02-28T191801.json" "refrac_corr_OVRO MMA_2024-02-28T193604.json" "refrac_corr_OVRO MMA_2024-02-28T194105.json" "refrac_corr_OVRO MMA_2024-02-28T195207.json" "refrac_corr_OVRO MMA_2024-02-28T195407.json" "refrac_corr_OVRO MMA_2024-02-28T200309.json" "refrac_corr_OVRO MMA_2024-02-28T201301.json" "refrac_corr_OVRO MMA_2024-02-28T201702.json" "refrac_corr_OVRO MMA_2024-02-28T202303.json" "refrac_corr_OVRO MMA_2024-02-28T203104.json" "refrac_corr_OVRO MMA_2024-02-28T205008.json" "refrac_corr_OVRO MMA_2024-02-28T205809.json"  --interp_method interp:cubic --timediff_tol 30
        python img_corr_pipeline.py --mode auto --freq 34 39 43 48 52 57 62 66 71 75 80 84 --norm log --level lev1 --filetype fits  --specmode mfs --workdir /Users/fisher/myworkspace --alpha 0.7 --minpercent 5 --fov 22000 22000 --dual_panel --get_latest_version --rfrcor_parm_files  "refrac_corr_OVRO MMA_2024-02-28T164502.json" "refrac_corr_OVRO MMA_2024-02-28T165003.json" "refrac_corr_OVRO MMA_2024-02-28T165604.json" "refrac_corr_OVRO MMA_2024-02-28T170035.json" "refrac_corr_OVRO MMA_2024-02-28T170556.json" "refrac_corr_OVRO MMA_2024-02-28T171207.json" "refrac_corr_OVRO MMA_2024-02-28T171418.json" "refrac_corr_OVRO MMA_2024-02-28T172009.json" "refrac_corr_OVRO MMA_2024-02-28T172800.json" "refrac_corr_OVRO MMA_2024-02-28T173201.json" "refrac_corr_OVRO MMA_2024-02-28T173602.json" "refrac_corr_OVRO MMA_2024-02-28T174604.json" "refrac_corr_OVRO MMA_2024-02-28T175706.json" "refrac_corr_OVRO MMA_2024-02-28T180607.json" "refrac_corr_OVRO MMA_2024-02-28T181209.json" "refrac_corr_OVRO MMA_2024-02-28T181800.json" "refrac_corr_OVRO MMA_2024-02-28T182000.json" "refrac_corr_OVRO MMA_2024-02-28T182501.json" "refrac_corr_OVRO MMA_2024-02-28T182701.json" "refrac_corr_OVRO MMA_2024-02-28T183102.json" "refrac_corr_OVRO MMA_2024-02-28T183302.json" "refrac_corr_OVRO MMA_2024-02-28T184004.json" "refrac_corr_OVRO MMA_2024-02-28T184805.json" "refrac_corr_OVRO MMA_2024-02-28T185707.json" "refrac_corr_OVRO MMA_2024-02-28T190308.json" "refrac_corr_OVRO MMA_2024-02-28T191009.json" "refrac_corr_OVRO MMA_2024-02-28T191200.json" "refrac_corr_OVRO MMA_2024-02-28T191801.json" "refrac_corr_OVRO MMA_2024-02-28T193604.json" "refrac_corr_OVRO MMA_2024-02-28T194105.json" "refrac_corr_OVRO MMA_2024-02-28T195207.json" "refrac_corr_OVRO MMA_2024-02-28T195407.json" "refrac_corr_OVRO MMA_2024-02-28T200309.json" "refrac_corr_OVRO MMA_2024-02-28T201301.json" "refrac_corr_OVRO MMA_2024-02-28T201702.json" "refrac_corr_OVRO MMA_2024-02-28T202303.json" "refrac_corr_OVRO MMA_2024-02-28T203104.json" "refrac_corr_OVRO MMA_2024-02-28T203505.json" "refrac_corr_OVRO MMA_2024-02-28T204206.json" "refrac_corr_OVRO MMA_2024-02-28T205008.json" "refrac_corr_OVRO MMA_2024-02-28T205809.json" "refrac_corr_OVRO MMA_2024-02-28T210601.json" "refrac_corr_OVRO MMA_2024-02-28T211502.json"  --interp_method interp:cubic fit:linear --timediff_tol 30
         
    '''
    parser = argparse.ArgumentParser(
        description="Process OVROLWA solar data with optional plotting and refraction correction.")
    parser.add_argument('--mode', type=str, choices=['auto', 'fast', 'slow'], default='auto',
                        help='Imaging mode [slow or fast]')
    parser.add_argument('--timestamp_file', type=str,
                        help='Path to a file containing timestamps, one per line in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--timestamps', type=str, nargs='+', required=False,
                        help='List of timestamps in YYYY-MM-DDTHH:MM:SS format')
    parser.add_argument('--start_time', type=str, help='Start time in YYYY-MM-DDTHH:MM:SS format')
    parser.add_argument('--end_time', type=str, help='End time in YYYY-MM-DDTHH:MM:SS format')
    parser.add_argument('--cadence', type=float, help='Cadence in seconds')
    parser.add_argument('--timediff_tol', type=float, default=None,
                        help='Time difference tolerance between the assigned timestamps and the file in seconds')
    parser.add_argument('--freq', type=float, nargs='+', default=[34, 43, 52, 62, 71, 84], help='Frequencies in MHz')
    parser.add_argument('--norm', type=str, choices=['linear', 'log'], default='log',
                        help='Normalization for the colormap')
    parser.add_argument('--level', type=str, default='lev1', help='Data level')
    parser.add_argument('--filetype', type=str, default='hdf', help='File type to download. Options: hdf, fits')
    parser.add_argument('--specmode', type=str, default='mfs', help='Image mode to download. Options: mfs, fch')
    parser.add_argument('--workdir', type=str, default='.', help='Path to refraction correction parameters')
    parser.add_argument('--datadir', type=str, default='.', help='Path to the data directory')
    parser.add_argument('--scaled_suvi', action='store_true', help='Scale the SUVI images')
    parser.add_argument('--suvi_passband', type=str, default='171', help='SUVI passband to use')
    parser.add_argument('--show_ax_grid', action='store_true', help='Show axis grid on the images')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha value for transparency')
    parser.add_argument('--snr_threshold', type=float, default=0.0, help='SNR threshold for the radio images')
    parser.add_argument('--minpercent', type=int, default=5,
                        help='Minimum value as a percentage of the data maximum for the colormap')
    parser.add_argument('--draw_contours', action='store_true', help='Draw contours on the images')
    parser.add_argument('--fov', type=int, nargs=2, default=[16000, 16000], help='Field of view in arcsec')
    parser.add_argument('--fov_center', type=int, nargs=2, default=[0, 0], help='Center of the field of view in arcsec')
    parser.add_argument('--dual_panel', action='store_true', help='Plot the radio and white-light images side by side')
    parser.add_argument('--autocorr', action='store_true',
                        help='Do automatic correlation for refraction. If this is set, the px1 and py1 will be overwritten by the calculated values.')
    parser.add_argument('--docorr', action='store_true', help='Do refraction correction')
    parser.add_argument('--get_latest_version', action='store_true',
                        help='Get the latest version of the parameters. Now we are using csv instead of json. The keyword will be obsolete in the next update.')
    parser.add_argument('--rfrcor_parm_files', type=str, nargs='+', default=[],
                        help='Refraction correction parameter files')
    parser.add_argument('--interp_method', type=str, nargs=2, default=['fit:linear', 'fit:linear'],
                        help='Interpolation methods for refraction correction parameters for pi1 and pi2 respectively (i stands for x and y). Options: mean, fit:linear, fit:quadratic, interp:linear, interp:nearest, interp:nearest-up, interp:zero, interp:quadratic, interp:cubic, interp:previous, interp:next. If one method is provided, it will be used for both pi1 and pi2.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing image jpg files')
    parser.add_argument('--trajectory_file', type=str, default='', help='Path to the trajectory file')
    parser.add_argument('--sshconfig', type=str, default='./ssh-to-data.private.config', help='SSH config file')
    parser.add_argument('--roi', type=float, nargs=4, action='append',
                        help='Define one or more regions of interest (ROIs) using bottom-left and top-right coordinates as (x1, y1, x2, y2). '
                             'To specify multiple ROIs, repeat the --roi option for each region. For example: '
                             '--roi 10 20 30 40 --roi 50 60 70 80 will define two ROIs.')
    parser.add_argument('--roi_percentile', type=float,
                        help='Percentile value for extracting the ROI spectrogram. If not specified, the ROI will default to the mean value.')
    args = parser.parse_args()

    if args.timestamps:
        timestamps = [datetime.fromisoformat(ts) for ts in args.timestamps]
    elif args.timestamp_file:
        timestamps = []
        with open(args.timestamp_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        timestamps.append(datetime.fromisoformat(line))
                    except ValueError:
                        print(f"Skipping unreadable timestamp: {line}")
        if not timestamps:
            raise ValueError("No valid timestamps found. Please check the input timestamp file.")
    elif args.start_time and args.end_time and args.cadence:
        start_time = datetime.fromisoformat(args.start_time)
        end_time = datetime.fromisoformat(args.end_time)
        cadence = timedelta(seconds=float(args.cadence))
        timestamps = [start_time + i * cadence for i in range(int((end_time - start_time) / cadence) + 1)]
    else:
        raise ValueError("Provide either --timestamps, --timestamp_file, or --start_time, --end_time, and --cadence")

    freqplts = args.freq * u.MHz
    mnorm = mcolors.LogNorm() if args.norm == 'log' else mcolors.Normalize()


    def resolve_path(file_path, base_dir):
        """
        Resolve file paths by checking under the current directory and the specified base directory.

        :param file_path: The path of the file to resolve.
        :param base_dir: The base directory to check if the file is not found in the current directory.
        :return: The absolute path if the file exists, otherwise returns None.
        """
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path  # Return absolute path if it exists
        elif os.path.exists(os.path.join(base_dir, file_path)):
            return os.path.abspath(os.path.join(base_dir, file_path))  # Check in the base directory
        elif os.path.exists(file_path):
            return os.path.abspath(file_path)  # Return path if it exists relative to the current directory
        return None


    # Assuming rfrcor_parm_files is initially a list of paths given in args
    resolved_rfrcor_parm_files = []
    for file_path in args.rfrcor_parm_files:
        resolved_path = resolve_path(file_path, args.workdir)
        if resolved_path:
            resolved_rfrcor_parm_files.append(resolved_path)
        else:
            print(f"Warning: Refraction correction parameter file not found: {file_path}")

    rfrcor_parm_files = resolved_rfrcor_parm_files

    # # Combine conversion to absolute paths and file existence check
    # rfrcor_parm_files = [
    #     os.path.abspath(f) if os.path.isabs(f) else os.path.join(args.workdir, f)
    #     for f in args.rfrcor_parm_files
    # ] if args.rfrcor_parm_files else []

    # Filter out non-existing files
    rfrcor_parm_files = [f for f in rfrcor_parm_files if os.path.exists(f)]

    trajectory_file = args.trajectory_file if os.path.exists(args.trajectory_file) else ''

    # Check if roi_percentile is provided as the string "None" and convert it to None
    if isinstance(args.roi_percentile, str) and args.roi_percentile.lower() == 'none':
        args.roi_percentile = None

    main(args.mode, timestamps, freqplts,
         level=args.level,
         filetype=args.filetype,
         specmode=args.specmode,
         mnorm=mnorm,
         refrac_corr=args.docorr,
         auto_corr=args.autocorr,
         workdir=args.workdir,
         datadir=args.datadir,
         scaled_suvi=args.scaled_suvi,
         suvi_passband=args.suvi_passband,
         show_ax_grid=args.show_ax_grid,
         alpha=args.alpha,
         snr_threshold=args.snr_threshold,
         minpercent=args.minpercent,
         draw_contours=args.draw_contours,
         fov=args.fov,
         fov_center=args.fov_center,
         dual_panel=args.dual_panel,
         roi=args.roi,
         roi_percentile=args.roi_percentile,
         get_latest_version=args.get_latest_version,
         timediff_tol=args.timediff_tol,
         rfrcor_parm_files=rfrcor_parm_files,
         interp_method=args.interp_method,
         trajectory_file=trajectory_file,
         overwrite=args.overwrite,
         sshconfig=args.sshconfig)
