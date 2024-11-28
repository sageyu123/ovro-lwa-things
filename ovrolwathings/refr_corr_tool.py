import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import sunpy.map as smap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIcon, QIntValidator
from PyQt5.QtWidgets import (QAbstractSlider, QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter,
                             QToolBar, QToolButton, QVBoxLayout, QWidget)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pyvistaqt import BackgroundPlotter
# import vtk
from skimage.feature import peak_local_max
from sunpy.coordinates import Helioprojective
from sunpy.map.maputils import all_coordinates_from_map

from ovrolwathings.utils import correct_images, define_filename, define_timestring, interpolate_rfrcorr_params, \
    norm_to_percent, plot_interpolated_params, pxy2shifts, shift2pxy

UI_dir = Path(__file__).parent / 'UI'
print(f"UI_dir: {UI_dir}")  

def enhance_offdisk_corona(sunpymap):
    hpc_coords = all_coordinates_from_map(sunpymap)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / sunpymap.rsun_obs
    rsun_step_size = 0.01
    rsun_array = np.arange(1, r.max(), rsun_step_size)
    y = np.array([sunpymap.data[(r > this_r) * (r < this_r + rsun_step_size)].mean()
                  for this_r in rsun_array])
    params = np.polyfit(rsun_array[rsun_array < 1.5],
                        np.log(y[rsun_array < 1.5]), 1)
    scale_factor = np.exp((r - 1) * (-params[0]))
    scale_factor[r < 1] = 1
    scaled_sunpymap = smap.Map(sunpymap.data * scale_factor, sunpymap.meta)
    return scaled_sunpymap


class FloatSlider(QSlider):
    valueChangedFloat = pyqtSignal(float)

    def __init__(self, orientation, parent=None, use_log_scale=True):
        super().__init__(orientation, parent)
        self.factor = 10000  # Factor to convert float to int
        self.use_log_scale = use_log_scale

    def setFloatRange(self, min_val, max_val):
        if self.use_log_scale:
            min_val = np.log10(min_val)
            max_val = np.log10(max_val)
        self.setMinimum(int(min_val * self.factor))
        self.setMaximum(int(max_val * self.factor))

    def setValue(self, value):
        if self.use_log_scale:
            if value == 0:
                ## this is to avoid log(0) error. the log(0) stems from the QDoubleValidator or QIntValidator. When the input text is out of range, the value is set to 0
                ## this is somewhat unexpected and the finding the bug took me a while.
                return
            # print(f"value: {value}")
            # print(
            #     f"value: {value}, log10(value): {np.log10(value)}, int(np.log10(value) * self.factor): {int(np.log10(value) * self.factor)} ")
            int_value = int(np.log10(value) * self.factor)
        else:
            int_value = int(value * self.factor)
        # print(f"slider value set to {int_value}, the min value is {self.minimum()}, the max value is {self.maximum()}")
        super().setValue(int_value)

    def value(self):
        float_value = super().value() / self.factor
        if self.use_log_scale:
            return 10 ** (float_value)
        return float_value

    def sliderChange(self, change):
        super().sliderChange(change)
        if change == QAbstractSlider.SliderValueChange:
            self.valueChangedFloat.emit(self.value())


class DelayedSliderWithInput(QWidget):
    def __init__(self, label_text, min_val=0, max_val=100, initial_val=50, delay_ms=200, float_slider=False,
                 use_log_scale=True):
        super().__init__()

        self.delay_ms = delay_ms
        self.updating_slider = False
        self.updating_input = False
        self.min_val = min_val
        self.max_val = max_val
        self.float_slider = float_slider
        if float_slider:
            self.use_log_scale = use_log_scale
        else:
            self.use_log_scale = False

        tooltip_text = f"Enter a value between {min_val} and {max_val}"
        self.label = QLabel(label_text)
        self.label.setToolTip(tooltip_text)

        if float_slider:
            self.slider = FloatSlider(Qt.Horizontal, use_log_scale=self.use_log_scale)
            self.slider.setFloatRange(min_val, max_val)
            self.slider.setValue(initial_val)
        else:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(min_val)
            self.slider.setMaximum(max_val)
            self.slider.setValue(initial_val)

        self.slider.valueChanged.connect(self.on_slider_value_changed)

        self.input_box = QLineEdit()
        self.input_box.setToolTip(tooltip_text)
        self.input_box_text_previous = str(initial_val)
        self.input_box.setText(self.input_box_text_previous)
        if float_slider:
            self.input_box.setValidator(QDoubleValidator(min_val, max_val, 1))
        else:
            self.input_box.setValidator(QIntValidator(min_val, max_val))
        self.input_box.returnPressed.connect(self.update_slider_from_input)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_input)

        # Create layout
        upper_row = QHBoxLayout()
        upper_row.addWidget(self.label)
        upper_row.addWidget(self.input_box)

        layout = QVBoxLayout()
        layout.addLayout(upper_row)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def on_slider_value_changed(self, value):
        if self.updating_slider:
            return
        self.timer.start(self.delay_ms)

    def set_input_text(self, value):
        if self.float_slider:
            self.input_box.setText(f"{value:.1f}")
        else:
            self.input_box.setText(str(value))

    def update_input(self):
        self.updating_input = True
        value = self.slider.value()
        self.set_input_text(value)
        self.updating_input = False

    def update_slider_from_input(self):
        if self.updating_input:
            return
        input_box_text_current = self.input_box.text()
        if self.float_slider:
            value = float(input_box_text_current)
        else:
            value = int(input_box_text_current)
        print(f"slider returned value: {self.slider.value()}")
        if value < self.min_val or value > self.max_val:
            print(
                f"value {value} out of range [{self.min_val}, {self.max_val}], resetting to previous value {self.input_box_text_previous}")
            self.input_box.setText(self.input_box_text_previous)
            return
        self.slider.setValue(value)
        self.input_box_text_previous = input_box_text_current


# Dummy function to generate a SunPy map for plotting
def create_smap(data, header, freqmhz):
    header['freq'] = freqmhz * 1e6
    header['frequnit'] = 'Hz'
    return smap.Map(data, header)


def validate_number(value, min_value=None, max_value=None, precision=None, inclusive=True):
    try:
        num = float(value)
        if inclusive:
            if (min_value is not None and num < min_value) or (max_value is not None and num > max_value):
                raise ValueError
        else:
            if (min_value is not None and num <= min_value) or (max_value is not None and num >= max_value):
                raise ValueError
        if precision is not None:
            return f"{num:.{precision}f}"
        return str(num)
    except ValueError:
        range_str = f"[{min_value}, {max_value}]" if inclusive else f"({min_value}, {max_value})"
        raise ValueError(f"Value '{value}' is not a valid number in the range {range_str}")


def validate_number_list(value, min_value=None, max_value=None, inclusive=True):
    try:
        numbers = list(map(float, value.rstrip(',').split(',')))
        valid_numbers = []
        for num in numbers:
            if inclusive:
                if min_value is not None and num < min_value:
                    valid_numbers.append(min_value)
                    continue  # Drop numbers below the minimum
                if max_value is not None and num > max_value:
                    valid_numbers.append(max_value)
                    continue  # Drop numbers above the maximum
                valid_numbers.append(num)
            else:
                if min_value is not None and num <= min_value:
                    valid_numbers.append(min_value)
                    continue
                if max_value is not None and num >= max_value:
                    valid_numbers.append(max_value)
                    continue
                valid_numbers.append(num)

        numbers = sorted(set(valid_numbers))
        if inclusive:
            if any((min_value is not None and num < min_value) or (max_value is not None and num > max_value) for num in
                   numbers):
                raise ValueError
        else:
            if any((min_value is not None and num <= min_value) or (max_value is not None and num >= max_value) for num
                   in numbers):
                print(f"here: min_value: {min_value}, max_value: {max_value}, numbers: {numbers}")
                raise ValueError

        # Format numbers: drop decimal if it's zero
        numbers = sorted(numbers)
        formatted_numbers = [str(int(num)) if num.is_integer() else str(num) for num in numbers]
        return ', '.join(formatted_numbers)
    except ValueError:
        range_str = f"[{min_value}, {max_value}]" if inclusive else f"({min_value}, {max_value})"
        raise ValueError(f"List '{value}' contains invalid numbers or values out of the range {range_str}")


# PyQt5 GUI setup
class ImageCorrectionApp(BackgroundPlotter):
    """
    A graphical user interface application for interactive image correction based on refraction effects in solar observations.

    This application allows users to visually adjust image alignment and apply corrections for refraction anomalies, commonly encountered in solar image data. The interface provides tools for specifying parameters for refraction correction, managing different solar observation frequencies, and viewing changes in real-time.

    :param image_series: np.ndarray
        The 3D numpy array containing the image series where each slice corresponds to a different frequency observation.
    :param meta: dict
        Metadata associated with the image series, typically including information such as headers and frequency details.
    :param background_map: list of sunpy.map.Map, optional
        A list of background solar images in various observational layers to be displayed behind the main image for contextual comparison. Each map must be an instance of `sunpy.map.Map`.

    Attributes
    ----------
    undo_stack : list
        A stack used to store the states for undo functionality.
    redo_stack : list
        A stack used for redo functionality, storing states that can be restored.
    cmap : matplotlib.colors.Colormap
        The colormap used for displaying images.
    header : dict
        The header information extracted from `meta` which contains metadata of the observation.
    cfreqs : np.ndarray
        An array of central frequencies associated with the image data.
    freqs : list
        List of frequencies selected for display and manipulation within the GUI.

    Methods
    -------
    initUI()
        Initializes the graphical components and layout of the interface.
    apply_correction()
        Applies the specified refraction corrections to the image series and updates the display.
    update_foreground()
        Updates the display properties and alignment of the foreground image based on user adjustments.
    save_params_to_file()
        Saves the current refraction correction parameters to a JSON file.
    load_params_from_file()
        Loads refraction correction parameters from a JSON file.
    """

    def __init__(self, image_series=None, meta=None, background_map=None, lwafile=None, freqs_default=None,
                 trajectory_file=None):
        super().__init__()
        # print(f"lwafile: {lwafile}, background_map: {background_map}")
        if lwafile:
            if lwafile.endswith('.hdf'):
                from ovrolwasolar.utils import recover_fits_from_h5
                self.meta, self.image_series = recover_fits_from_h5(lwafile, return_data=True)
            else:
                from suncasa.io import ndfits
                self.meta, self.image_series = ndfits.read(lwafile)
                if 'cfreqs' not in self.meta:
                    self.meta['cfreqs'] = self.meta['ref_cfreqs']

            self.image_series = np.squeeze(self.image_series)
        else:
            self.image_series = np.squeeze(image_series)
            self.meta = meta

        if trajectory_file:
            import pickle
            with open(trajectory_file, 'rb') as f:
                self.trajs = pickle.load(f)
        else:
            self.trajs = None

        # Initialize undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.updating_contour_plot_input = False
        self.cmap = plt.get_cmap('jet')
        self.opacity_contours = 0.5
        self.opacity_rimg = 0.5
        self.contour_line_width = 2.0
        self.header = self.meta['header']
        self.delta_x = self.header['CDELT1']
        self.delta_y = self.header['CDELT2']
        self.zoffset = 0.001
        self.cfreqs = self.meta['cfreqs']
        self.cfreqsmhz = self.cfreqs / 1e6
        self.cfreqsmhzmin = np.min(self.cfreqsmhz)
        self.cfreqsmhzmax = np.max(self.cfreqsmhz)
        if freqs_default is None:
            self.freqs_default = "34, 43, 52, 62, 71, 84"
        else:
            self.freqs_default = freqs_default
        self.freqs = None
        self.rmap = create_smap(self.image_series[-1], self.header, self.cfreqs[-1])
        self.rmap_origin = None
        # self.params_filename = define_filename(self.rmap, prefix="refrac_corr_", ext=".json", get_latest_version=True)
        self.timestamp = define_timestring(self.rmap)
        self.time = self.rmap.date
        self.rsun = self.rmap.rsun_obs.to(u.arcsec).value
        self.top_right = SkyCoord(6000 * u.arcsec, 6000 * u.arcsec, frame=self.rmap.coordinate_frame)
        self.bottom_left = SkyCoord(-6000 * u.arcsec, -6000 * u.arcsec, frame=self.rmap.coordinate_frame)
        peak_coords = peak_local_max(self.rmap.data, min_distance=60, threshold_rel=0.2)
        hpc_max = self.rmap.wcs.pixel_to_world(peak_coords[:, 1] * u.pixel, peak_coords[:, 0] * u.pixel)[0]
        self.frg_sphere_origin = (hpc_max.Tx.value, 0, hpc_max.Ty.value)
        self.background_map = {}
        self.contours_paths = []
        self.plane_main = []
        self.plane_background = None
        self.bkg_sphere = None
        self.frg_sphere = None
        self.save_state_flag = False  # Flag to control state saving
        self.df_params = None
        self.fig_params = None
        self.sc_px0 = None
        self.sc_py0 = None
        self.sc_px1 = None
        self.sc_py1 = None

        if background_map is not None:
            self.zoffset = 0.001 * np.arange(len(background_map), 0, -1)
            _ = self.init_background_map(background_map)
        self.px = [0.0, 0.0]
        self.py = [0.0, 0.0]
        self.update_frg_sphere_input_fields = True
        self.update_bkg_sphere_input_fields = True

        # Store previous valid values for input fields
        self.previous_values = {
            "px0": 0.0,
            "py0": 0.0,
            "px1": 0.0,
            "py1": 0.0,
            "freq": "34, 43, 52, 62, 71, 84",
            "contourlevels": "5, 15, 30, 60, 90"
        }
        self.initUI()
        self.init_background()
        self.init_trajs()
        self.view_xz()
        self.camera.ParallelProjectionOn()
        self.save_state_flag = True  # Flag to control state saving

    def get_bkgmap_name(self, bmap):
        instr = bmap.instrument if hasattr(bmap, 'instrument') else None
        det = bmap.detector if hasattr(bmap, 'detector') else None
        wav = bmap.wavelength if hasattr(bmap, 'wavelength') else None
        if wav is not None:
            wav = f"{wav.to(u.AA).value:.0f}"
        bmapname = "_".join(bmap for bmap in [instr, det, wav] if bmap is not None)
        return bmapname

    def init_background_map(self, background_map, reprojection=False):
        # print("step 1")
        if isinstance(background_map, smap.mapbase.GenericMap):
            self.background_map = [background_map]
            return
        if isinstance(background_map, (str, Path)):
            background_map = [background_map]
        if all(isinstance(x, (str, Path)) for x in background_map):  # If all elements are file paths
            bkg_map = {}
            for x in background_map:
                bmap = smap.Map(x)
                bmapname = self.get_bkgmap_name(bmap)
                if bmapname.startswith('SUVI'):
                    bmap = enhance_offdisk_corona(bmap)
                bkg_map[bmapname] = bmap
            background_map = bkg_map
        elif all(isinstance(x, smap.mapbase.GenericMap) for x in
                 background_map):  # If all elements are already Map instances
            bkg_map = {}
            for x in background_map:
                bmapname = self.get_bkgmap_name(x)
                bkg_map[bmapname] = x
            background_map = bkg_map
        else:
            raise ValueError("background_map must be a list of file paths or a list of sunpy.map.Map instances.")
        #         print("step 2")
        self.background_map = background_map
        projected_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                                   obstime=self.rmap.observer_coordinate.obstime,
                                   frame='helioprojective',
                                   observer=self.rmap.observer_coordinate,
                                   rsun=self.rmap.coordinate_frame.rsun)
        #         print("step 3")

        if reprojection:
            self.background_map = {}
            for bkg_map_nama, bkg_map in background_map.items():
                projected_header = smap.make_fitswcs_header(bkg_map.data.shape,
                                                            projected_coord,
                                                            scale=u.Quantity(bkg_map.scale),
                                                            instrument=bkg_map.instrument,
                                                            wavelength=bkg_map.wavelength)
                with Helioprojective.assume_spherical_screen(bkg_map.observer_coordinate):
                    bkg_map_projected = bkg_map.reproject_to(projected_header)
                    self.background_map[bkg_map_nama] = bkg_map_projected
        else:
            self.background_map = background_map
        return

    def map_prep(self, sunmap, zoffset=0.0, add_freq_axis=False):
        """
        Prepare SunPy map for plotting.

        :param sunmap: sunpy.map.Map
            The SunPy map object to be prepared.
        :param zoffset: float, optional
            Offset along the z-axis, by default 0.0.
        :return: tuple
            Tuple containing data, spacing, origin, dimensions, and rsun.
        """

        header = sunmap.meta
        data = sunmap.data
        ny, nx = data.shape
        dx, dy = sunmap.scale
        dx = dx.to(u.arcsec / u.pix).value
        dy = dy.to(u.arcsec / u.pix).value
        rsun = sunmap.rsun_obs.to(u.arcsec).value
        spacing = (dx, (dx + dy) / 2.0, dy)
        if add_freq_axis:
            reffreqmhz = (header['freq'] * u.Unit(header['frequnit'])).to(u.MHz).value
            domin_size = np.sqrt((nx * dx) ** 2 + (ny * dy) ** 2)
            print(
                f'domin_size: {domin_size}, reffreqmhz: {reffreqmhz}, cfreqsmhzmin: {self.cfreqsmhzmin}, cfreqsmhzmax: {self.cfreqsmhzmax}')
            zoffset_freq = 0.25 * domin_size * (reffreqmhz - self.cfreqsmhzmax) / (
                    self.cfreqsmhzmax - self.cfreqsmhzmin)
        else:
            zoffset_freq = 0.0
        origin = (
            header['CRVAL1'] - (header['CRPIX1'] - 1) * dx, zoffset + zoffset_freq,
            header['CRVAL2'] - (header['CRPIX2'] - 1) * dy)
        dimensions = (nx, 1, ny)
        data = data.T.ravel(order="F")
        return data, spacing, origin, dimensions, rsun

    def update_plane_data(self, sunpy_map, freq, color, add_dslice=False):
        """
        Update plane data from SunPy map.

        :param sunpy_map: sunpy.map.Map
            SunPy map object.
        :param freq: float
            Frequency of the image.
        :param color: tuple
            Color for the contour lines.
        :return: dict
            Dictionary containing the updated plane data.
        """

        sunpy_map_crop = sunpy_map.submap(self.bottom_left, top_right=self.top_right)
        data, spacing, origin, dimensions, rsun = self.map_prep(sunpy_map_crop, add_freq_axis=True)
        img_data = self.create_pvimgdata(data, spacing, origin, dimensions)

        actor_rimg = None
        actor_rdslice = None
        conts = img_data.contour(isosurfaces=self.levels / 100 * np.nanmax(data), scalars=img_data['scalar'])
        actor_contour = self.add_mesh(conts, color=color, opacity=self.opacity_contours, rgb=False,
                                      render_lines_as_tubes=True, line_width=self.contour_line_width,
                                      show_scalar_bar=False)

        return {'img_data': img_data, 'actor_rimg': actor_rimg, 'actor_rdslice': actor_rdslice,
                'actor_contour': actor_contour, 'cmap': "viridis",
                'origin': img_data.origin, 'freq': freq}

    def create_pvimgdata(self, data, spacing, origin, dimensions):
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
        img_data['dslice_data'] = np.zeros(data.shape)
        return img_data

    def init_trajs(self):
        self.trajs_actors = None
        if self.trajs is not None:
            self.trajs_actors = {'actor_cut': None, 'line_cut': None, 'tube_cut': None, 'points_cut': None,
                                 'actor_trajs': [], 'point_trajs': [], 'sphere_trajs': []}
            from scipy.interpolate import interp1d
            method = 'linear'
            cutslit = self.trajs['cutslit']
            points = np.vstack([cutslit['xcen'], [0] * len(cutslit['xcen']), cutslit['ycen']]).T
            # actor = self.add_lines(points, color='purple', width=5, connected=True)
            # Create a line from points
            line = pv.lines_from_points(points)
            # Create a tube around the line
            tube = line.tube(radius=10)
            # Add the tube to the plotter
            actor = self.add_mesh(tube, color='purple', opacity=0.5)
            self.trajs_actors['actor_cut'] = actor
            self.trajs_actors['points_cut'] = points
            self.trajs_actors['line_cut'] = line
            self.trajs_actors['tube_cut'] = tube

            xcen = cutslit['xcen']
            ycen = cutslit['ycen']
            dist = cutslit['dist']

            interp_func_xcen = interp1d(dist, xcen, kind=method, fill_value="extrapolate")
            interp_func_ycen = interp1d(dist, ycen, kind=method, fill_value="extrapolate")

            tplt_now = self.time.plot_date
            for traj in self.trajs['trajs']:
                trajdist = traj['dist']
                trajtplt = traj['tplt']
                interp_func_dist = interp1d(trajtplt, trajdist, kind=method, fill_value="extrapolate")
                dist_now = interp_func_dist(tplt_now)
                xcen_now = interp_func_xcen(dist_now)
                ycen_now = interp_func_ycen(dist_now)

                point = (xcen_now.item(), 0, ycen_now.item(),)
                point = np.array(point)
                sphere = pv.Sphere(radius=self.rsun * 0.1, center=point, theta_resolution=30, phi_resolution=30)

                # Add the sphere to the plotter with transparency
                actor = self.add_mesh(sphere, color='red', opacity=0.5)
                self.trajs_actors['actor_trajs'].append(actor)
                self.trajs_actors['point_trajs'].append(point)
                # self.trajs_actors['sphere_trajs'].append(sphere)

    def update_trajs(self, center=(0, 0, 0)):
        if self.trajs_actors is not None:
            points_new = self.trajs_actors['points_cut'] + np.array(center).T[np.newaxis, :]
            self.trajs_actors['line_cut'].points = points_new
            self.trajs_actors['tube_cut'] = self.trajs_actors['line_cut'].tube(radius=10)
            self.trajs_actors['actor_cut'].GetMapper().SetInputData(self.trajs_actors['tube_cut'])

            # Update spheres
            for idx, point in enumerate(self.trajs_actors['point_trajs']):
                point_new = point + np.array(center)
                sphere = pv.Sphere(radius=self.rsun * 0.1, center=point_new, theta_resolution=30, phi_resolution=30)
                # self.trajs_actors['sphere_trajs'][idx] = sphere
                self.trajs_actors['actor_trajs'][idx].GetMapper().SetInputData(sphere)

            self.render()

    def init_background(self):
        """
        Initialize the background image data and actors.
        """
        self.plane_background = []
        for idx, (bkgmapname, bkg_map) in enumerate(self.background_map.items()):
            data, spacing, origin, dimensions, rsun = self.map_prep(bkg_map, zoffset=self.zoffset[idx])
            img_data = pv.ImageData()
            img_data.spacing = spacing
            img_data.origin = origin
            img_data.dimensions = dimensions
            img_data['scalar'] = data
            # cmap = bkg_map.cmap
            cmap = plt.get_cmap('gray_r')
            # print(f"init bkgmapname: {bkgmapname}, origin: {origin}")

            mask = data == 0
            # Create an opacity array where NaNs are transparent (0 opacity)
            opacity_array = np.ones(data.shape)
            opacity_array[mask] = 0.5
            img_data['opacity'] = opacity_array.ravel(order='F')

            actor = self.add_mesh(img_data, scalars='scalar', rgb=False,
                                  cmap=cmap, opacity='opacity',
                                  clim=(0, 255),
                                  show_edges=False,
                                  preference='cell',
                                  specular = 1.0,
                                  pickable=False, show_scalar_bar=False)
            self.plane_background.append(
                {'bkgmapname': bkgmapname, 'img_data': img_data, 'actor': actor, 'cmap': cmap, 'origin': origin})

        self.bkg_sphere = self.add_sphere_widget(self._on_bkg_sphere_move, center=(0, 0, 0), radius=self.rsun,
                                                 theta_resolution=20,
                                                 phi_resolution=20, style='wireframe')
        self.px0_input.returnPressed.connect(self.update_foreground)
        self.py0_input.returnPressed.connect(self.update_foreground)
        self.px1_input.returnPressed.connect(self.move_bkg_sphere)
        self.py1_input.returnPressed.connect(self.move_bkg_sphere)
        self.reset_rfr_parms_button.clicked.connect(self.reset_rfr_corr_params)
        self.load_rfr_parms_button.clicked.connect(self.load_params_from_file)
        self.save_rfr_parms_button.clicked.connect(self.save_params_to_file)
        self.redo_rfr_parms_button.clicked.connect(self.redo_rfr_parms)
        self.undo_rfr_parms_button.clicked.connect(self.undo_rfr_parms)
        self.show_rimg_checkbox.setChecked(False)
        self.toggle_radio_image_visibility(False)

    def init_foreground(self):
        """
        Initialize the foreground image data and actors.
        """
        self.freqs = list(map(float, self.freq_input.text().split(',')))
        self.freqs = np.sort(self.freqs)
        colors = self.cmap(np.linspace(0, 1, len(self.freqs)))
        nfreqs = len(self.freqs)
        for idx, freq in enumerate(self.freqs):
            fidx = np.argmin(np.abs(self.cfreqsmhz - freq))
            sunpy_map = create_smap(self.image_series[fidx], self.header, freq)
            plane_main_data = self.update_plane_data(sunpy_map, freq, colors[idx])
            if idx == 0:
                self.rmap_origin = plane_main_data['origin']
            self.plane_main.append(plane_main_data)
            if idx == nfreqs - 1:
                self.frg_sphere = self.add_sphere_widget(self._on_frg_sphere_move,
                                                         center=self.frg_sphere_origin,
                                                         radius=self.rsun * 0.2,
                                                         theta_resolution=20,
                                                         phi_resolution=20,
                                                         style='wireframe')

    def update_shifts(self, center):
        """
        Compute and update the shifts from the sphere's center.

        :param center: tuple
            Center coordinates of the sphere in arcsec.
        :return: tuple
            Computed shifts (px, py).
        """
        px, py = shift2pxy(center[0] - self.frg_sphere_origin[0], center[2] - self.frg_sphere_origin[2],
                           self.cfreqsmhz[-1])
        self.px[0] = px
        self.py[0] = py
        self.shifts_x, self.shifts_y = pxy2shifts(self.px, self.py, self.freqs)
        # print(self.shifts_x, self.shifts_y)
        return px, py

    def _on_frg_sphere_move(self, center):
        """
        Callback for foreground sphere movement.

        :param center: tuple
            New center coordinates of the sphere.
        """
        if self.frg_sphere is None:
            return
        center = (center[0], 0, center[2])
        self.frg_sphere.SetCenter(center)
        px, py = self.update_shifts(center)
        if self.update_frg_sphere_input_fields:
            self.px0_input.setText(f"{-px:.3e}")
            self.py0_input.setText(f"{-py:.3e}")
        self.update_foreground()
        self.update_params_plot()

    def move_frg_sphere(self, center=None):
        """
        Move the foreground sphere to a new position.

        :param center: tuple, optional
            New center coordinates for the sphere, by default None.
        """
        if center is None:
            px = -float(self.px0_input.text())
            py = -float(self.py0_input.text())
            self.px[0] = px
            self.py[0] = py
            shift_x, shift_y = pxy2shifts(self.px, self.py, self.cfreqsmhz[-1])
            center = (self.frg_sphere_origin[0] + shift_x, 0, self.frg_sphere_origin[2] + shift_y)
        self.update_frg_sphere_input_fields = False
        self._on_frg_sphere_move(center)
        self.update_frg_sphere_input_fields = True

    def reset_rfr_corr_params(self):
        """
        Reset refraction correction parameters to default values.
        """
        self.px0_input.setText("0.0")
        self.py0_input.setText("0.0")
        self.px1_input.setText("0.0")
        self.py1_input.setText("0.0")
        # self.save_state_flag = False
        self.move_bkg_sphere((0, 0, 0))
        self.move_frg_sphere(self.frg_sphere_origin)
        # self.save_state_flag = True

    def move_bkg_sphere(self, center=None):
        """
        Move the background sphere to a new position.

        :param center: tuple, optional
            New center coordinates for the sphere, by default None.
        """
        if center is None:
            try:
                x = float(validate_number(self.px1_input.text()))
                y = float(validate_number(self.py1_input.text()))
                self.previous_values['px1'] = self.px1_input.text()
                self.previous_values['py1'] = self.py1_input.text()
            except ValueError:
                self.px1_input.setText(self.previous_values['px1'])
                self.py1_input.setText(self.previous_values['py1'])
                return
            center = (x, 0, y)
        self.update_bkg_sphere_input_fields = False
        self._on_bkg_sphere_move(center)
        self.update_bkg_sphere_input_fields = True

    def _on_bkg_sphere_move(self, center):
        """
        Callback for background sphere movement.

        :param center: tuple
            New center coordinates of the sphere.
        """
        if self.bkg_sphere is None:
            return
        self.bkg_sphere.SetCenter(center)
        if self.update_bkg_sphere_input_fields:
            self.px1_input.setText(f"{center[0]:.2f}")
            self.py1_input.setText(f"{center[2]:.2f}")
        self.update_background()
        self.update_trajs(center)
        self.update_params_plot()

    def update_background(self):
        """
        Update the background image data and actors.
        """
        # Save current state before making changes, only if the flag is set
        if self.save_state_flag:
            self.save_rfr_parms_state()

        try:
            self.px1_input.setText(validate_number(self.px1_input.text()))
            self.py1_input.setText(validate_number(self.py1_input.text()))
            self.previous_values['px1'] = self.px1_input.text()
            self.previous_values['py1'] = self.py1_input.text()
        except ValueError:
            self.px1_input.setText(self.previous_values['px1'])
            self.py1_input.setText(self.previous_values['py1'])

        center = (float(self.px1_input.text()), 0, float(self.py1_input.text()))
        for idx, plane_bkg in enumerate(self.plane_background):
            sph_origin = plane_bkg['origin']
            # print(f"bkgmapname: {plane_bkg['bkgmapname']}, origin: {sph_origin}")
            plane_bkg['img_data'].origin = (center[0] + sph_origin[0], self.zoffset[idx], center[2] + sph_origin[2])

    def toggle_radio_contours_visibility(self, checked):
        """
        Toggle the visibility of the contours.

        :param checked: bool
            Whether the contours should be visible.
        """
        colors = self.cmap(np.linspace(0, 1, len(self.plane_main)))
        for idx, plane in enumerate(self.plane_main):
            if checked:
                if plane['actor_contour'] is None:
                    img_data = plane['img_data']
                    conts = img_data.contour(isosurfaces=self.levels / 100 * np.nanmax(img_data['scalar']),
                                             scalars=img_data['scalar'])
                    plane['actor_contour'] = self.add_mesh(conts, color=colors[idx], opacity=self.opacity_contours,
                                                           rgb=False,
                                                           render_lines_as_tubes=True,
                                                           line_width=self.contour_line_width,
                                                           show_scalar_bar=False)
            else:
                self.remove_actor(plane['actor_contour'])
                plane['actor_contour'] = None

    def toggle_radio_image_visibility(self, checked):
        for plane in self.plane_main:
            if checked:
                if plane['actor_rimg'] is None:
                    plane['actor_rimg'] = self.add_mesh(plane['img_data'], cmap="viridis", opacity=self.opacity_rimg,
                                                        rgb=False,
                                                        show_scalar_bar=False, pickable=False)
            else:
                self.remove_actor(plane['actor_rimg'])
                plane['actor_rimg'] = None

    def toggle_density_slice_visibility(self, checked):
        """
        Toggle the visibility of the pseudo-colored images.

        :param checked: bool
            Whether the pseudo-colored images should be visible.
        """
        colors_frac = np.linspace(0, 1, len(self.plane_main))
        for idx, plane in enumerate(self.plane_main):
            if checked:
                self.show_rimg_checkbox.setChecked(False)
                if plane['actor_rdslice'] is None:
                    img_data = plane['img_data']
                    data_dslice = np.ones_like(img_data['scalar']) * colors_frac[idx] * 255
                    img_data['dslice_data'] = data_dslice
                    opacity_array = norm_to_percent(img_data['scalar'].ravel(order='F'),
                                                    minpercent=np.nanmin(self.levels))
                    plane['actor_rdslice'] = self.add_mesh(img_data, scalars='dslice_data', cmap=self.cmap,
                                                           clim=[0, 255],
                                                           opacity=opacity_array, rgb=False,
                                                           show_scalar_bar=False, pickable=False)
            else:
                self.remove_actor(plane['actor_rdslice'])
                plane['actor_rdslice'] = None

    def initUI(self):
        timestamp_layout = QVBoxLayout()
        self.timestamp_label1 = QLabel(f"Data: {self.timestamp.split(' ')[0]}")
        self.timestamp_label2 = QLabel(f"Time: {' '.join(self.timestamp.split(' ')[1:])}")
        timestamp_layout.addWidget(self.timestamp_label1)
        timestamp_layout.addWidget(self.timestamp_label2)

        # Refraction Correction GroupBox
        rfr_corr_groupbox = QGroupBox("Refraction Correction")
        rfr_corr_layout = QVBoxLayout()

        rfr_corr_groupbox.setToolTip("Move the spheres to adjust the refraction correction parameters")
        px0_layout = QHBoxLayout()
        self.px0_label = QLabel('σₓ [arcsec/Hz²]:')
        self.px0_label.setToolTip(r'1/f² dependent coefficient in X (RFRPX0)')
        self.px0_input = QLineEdit(self)
        self.px0_input.setText("0.0")
        self.px0_input.setToolTip(r'1/f² dependent coefficient in X (RFRPX0)')
        px1_layout = QHBoxLayout()
        self.px1_label = QLabel('X shift [arcsec]:')
        self.px1_label.setToolTip('Absolute shift in solar X direction (RFRPX1)')
        self.px1_input = QLineEdit(self)
        self.px1_input.setToolTip('Absolute shift in solar X direction  (RFRPX1)')
        self.px1_input.setText("0.0")
        px0_layout.addWidget(self.px0_label)
        px0_layout.addWidget(self.px0_input)
        px1_layout.addWidget(self.px1_label)
        px1_layout.addWidget(self.px1_input)

        px_layout = QVBoxLayout()
        px_layout.addLayout(px0_layout)
        px_layout.addLayout(px1_layout)

        py0_layout = QHBoxLayout()
        self.py0_label = QLabel('σᵧ [arcsec/Hz²]:')
        self.py0_label.setToolTip('1/f² dependent coefficient in Y (RFRPY0)')
        self.py0_input = QLineEdit(self)
        self.py0_input.setText("0.0")
        self.py0_input.setToolTip('1/f² dependent coefficient in Y (RFRPY0)')
        py1_layout = QHBoxLayout()
        self.py1_label = QLabel('Y shift [arcsec]:')
        self.py1_label.setToolTip('Absolute shift in solar Y direction  (RFRPY1)')
        self.py1_input = QLineEdit(self)
        self.py1_input.setToolTip('Absolute shift in solar Y direction  (RFRPY1)')
        self.py1_input.setText("0.0")
        py0_layout.addWidget(self.py0_label)
        py0_layout.addWidget(self.py0_input)
        py1_layout.addWidget(self.py1_label)
        py1_layout.addWidget(self.py1_input)

        py_layout = QVBoxLayout()
        py_layout.addLayout(py0_layout)
        py_layout.addLayout(py1_layout)

        rfr_parms_ctrl_layout1 = QHBoxLayout()
        # rfr_parms_ctrl_layout2 = QHBoxLayout()

        # Replace button labels with symbols
        self.undo_rfr_parms_button = QPushButton('', self)
        self.undo_rfr_parms_button.setIcon(QIcon(str(UI_dir / 'undo_icon.png')))
        self.undo_rfr_parms_button.setToolTip('Undo')
        rfr_parms_ctrl_layout1.addWidget(self.undo_rfr_parms_button)

        self.redo_rfr_parms_button = QPushButton('', self)
        self.redo_rfr_parms_button.setIcon(QIcon(str(UI_dir / 'redo_icon.png')))
        self.redo_rfr_parms_button.setToolTip('Redo')
        rfr_parms_ctrl_layout1.addWidget(self.redo_rfr_parms_button)

        self.reset_rfr_parms_button = QPushButton('', self)
        self.reset_rfr_parms_button.setIcon(QIcon(str(UI_dir / 'reset_icon.png')))
        self.reset_rfr_parms_button.setToolTip('Reset parameters to Default')
        rfr_parms_ctrl_layout1.addWidget(self.reset_rfr_parms_button)
        # rfr_parms_ctrl_layout1.addStretch()

        self.load_rfr_parms_button = QPushButton('', self)
        self.load_rfr_parms_button.setIcon(QIcon(str(UI_dir / 'open_icon.png')))
        self.load_rfr_parms_button.setToolTip('Load parameters from a CSV file')
        rfr_parms_ctrl_layout1.addWidget(self.load_rfr_parms_button)

        self.save_rfr_parms_button = QPushButton('', self)
        self.save_rfr_parms_button.setIcon(QIcon(str(UI_dir / 'save_icon.png')))
        self.save_rfr_parms_button.setToolTip('Save parameters to a CSV file')
        rfr_parms_ctrl_layout1.addWidget(self.save_rfr_parms_button)
        rfr_parms_ctrl_layout1.addStretch()

        rfr_corr_layout.addLayout(px_layout)
        rfr_corr_layout.addLayout(py_layout)
        rfr_corr_layout.addLayout(rfr_parms_ctrl_layout1)
        # rfr_corr_layout.addLayout(rfr_parms_ctrl_layout2)
        rfr_corr_groupbox.setLayout(rfr_corr_layout)

        # Plot Options GroupBox
        plot_options_groupbox = QGroupBox("Plot Options")
        plot_options_layout = QVBoxLayout()

        self.show_rcontours_checkbox = QCheckBox('Contours', self)
        self.show_rcontours_checkbox.setChecked(True)
        self.show_rcontours_checkbox.toggled.connect(self.toggle_radio_contours_visibility)

        self.show_rimg_checkbox = QCheckBox('Image', self)
        self.show_rimg_checkbox.setChecked(False)
        self.show_rimg_checkbox.toggled.connect(self.toggle_radio_image_visibility)

        self.show_densityslice_checkbox = QCheckBox('Density Slice', self)
        self.show_densityslice_checkbox.setChecked(False)
        self.show_densityslice_checkbox.toggled.connect(self.toggle_density_slice_visibility)

        content_layout = QVBoxLayout()
        content_layout.addWidget(self.show_rcontours_checkbox)
        content_layout.addWidget(self.show_rimg_checkbox)
        content_layout.addWidget(self.show_densityslice_checkbox)

        plot_options_layout.addLayout(content_layout)

        freq_layout = QVBoxLayout()
        freq_count_layout = QHBoxLayout()
        freq_input_layout = QHBoxLayout()
        self.freq_label = QLabel('Frequencies [MHz]:')
        self.freq_label.setToolTip(
            f'Comma-separated list of frequencies. Enter values in the range of {self.cfreqsmhz[0]:.3f} to {self.cfreqsmhz[-1]:.3f} MHz')
        self.freq_input = QLineEdit(self)
        self.freq_input.setToolTip(
            f'Comma-separated list of frequencies. Enter values in the range of {self.cfreqsmhz[0]:.3f} to {self.cfreqsmhz[-1]:.3f} MHz')
        self.freq_input.setText(self.freqs_default)
        self.freq_input.returnPressed.connect(self.validate_and_update_contours)
        self.freq_reset = QPushButton('', self)
        self.freq_reset.setIcon(QIcon(str(UI_dir / 'reset_icon.png')))
        self.freq_reset.setMaximumWidth(40)
        self.freq_reset.setToolTip('Reset Frequencies to Default')
        self.freq_reset.clicked.connect(self.reset_freqs_to_default)

        # Number of Frequencies SpinBox
        self.freq_count_spinbox = QSpinBox(self)
        self.freq_count_spinbox.setToolTip('Number of frequencies to display')
        self.freq_count_spinbox.setRange(1, len(self.cfreqsmhz))
        self.freq_count_spinbox.setValue(6)
        self.freq_count_spinbox.setMaximumWidth(50)
        self.freq_count_spinbox.valueChanged.connect(self.update_freqs_from_spinbox)

        freq_count_layout.addWidget(self.freq_label)
        freq_count_layout.addWidget(self.freq_count_spinbox)
        freq_input_layout.addWidget(self.freq_input)
        freq_input_layout.addWidget(self.freq_reset)
        freq_layout.addLayout(freq_count_layout)
        freq_layout.addLayout(freq_input_layout)

        contour_layout = QVBoxLayout()
        self.contourlevels_label = QLabel('Contour Levels [%]:')
        self.contourlevels_label.setToolTip(
            'Comma-separated list of contour levels in percentage of the maximum intensity')
        self.contourlevels_input = QLineEdit(self)
        self.contourlevels_input.setToolTip(
            'Comma-separated list of contour levels in percentage of the maximum intensity')
        self.contourlevels_input.setText("10, 30, 60, 90")
        self.contourlevels_input.returnPressed.connect(self.validate_and_update_contours)
        contour_layout.addWidget(self.contourlevels_label)
        contour_layout.addWidget(self.contourlevels_input)

        plot_options_layout.addLayout(freq_layout)
        plot_options_layout.addLayout(contour_layout)

        self.opacity_control = DelayedSliderWithInput(
            label_text='Opacity [%]:',
            min_val=0,
            max_val=100,
            initial_val=int(self.opacity_contours * 100),
            delay_ms=20
        )
        # Set the size policy to ensure it expands correctly within the layout
        self.opacity_control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.opacity_control.slider.valueChanged.connect(self.update_contour_opacity_from_slider)
        self.opacity_control.input_box.returnPressed.connect(self.update_contour_opacity_from_input)

        plot_options_layout.addWidget(self.opacity_control)

        self.contour_lw_control = DelayedSliderWithInput(
            label_text='Line Width:',
            min_val=0.2,
            max_val=200,
            initial_val=self.contour_line_width,
            delay_ms=20,
            float_slider=True
        )
        # Set the size policy to ensure it expands correctly within the layout
        self.contour_lw_control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.contour_lw_control.slider.valueChangedFloat.connect(self.update_contour_lw_from_slider)
        self.contour_lw_control.input_box.returnPressed.connect(self.update_contour_lw_from_input)

        plot_options_layout.addWidget(self.contour_lw_control)

        self.y_scale_control = DelayedSliderWithInput(
            label_text='Z Scale:',
            min_val=0.2,
            max_val=5,
            initial_val=1,
            delay_ms=20,
            float_slider=True
        )
        # Ensure the new control expands correctly within the layout
        self.y_scale_control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Connect the y scale control to the update function
        self.y_scale_control.slider.valueChangedFloat.connect(self.update_y_scale)

        # Add the y scale control to the plot options layout
        plot_options_layout.addWidget(self.y_scale_control)

        plot_options_groupbox.setLayout(plot_options_layout)

        # Apply button
        self.apply_button = QPushButton('Apply Correction', self)
        self.apply_button.clicked.connect(self.apply_correction)

        # Main layout
        self.main_ctrl_layout = QHBoxLayout()
        self.main_ctrl_layout1 = QVBoxLayout()
        self.main_ctrl_layout1.addLayout(timestamp_layout)
        self.main_ctrl_layout1.addWidget(rfr_corr_groupbox)
        self.main_ctrl_layout1.addWidget(plot_options_groupbox)
        self.main_ctrl_layout1.addWidget(self.apply_button)
        self.main_ctrl_layout.addLayout(self.main_ctrl_layout1)

        self.main_ctrl_layout2 = QVBoxLayout()
        self.main_ctrl_layout.addLayout(self.main_ctrl_layout2)

        # Create a new horizontal layout with QSplitter for resizability
        central_widget = self.app_window.centralWidget()
        self.central_layout = central_widget.layout()

        last_widget = self.central_layout.itemAt(self.central_layout.count() - 1).widget()
        self.central_layout.removeWidget(last_widget)

        # Create a new horizontal layout with splitter
        splitter = QSplitter()

        # Add the last widget to the splitter
        splitter.addWidget(last_widget)

        # Add the main control layout to the splitter
        main_ctrl_widget = QWidget()
        main_ctrl_widget.setLayout(self.main_ctrl_layout)
        main_ctrl_widget.setMaximumWidth(250)
        splitter.addWidget(main_ctrl_widget)

        # Set the initial sizes (half size for main control layout)
        splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.25)])

        # Set the new layout to the central widget
        self.central_layout.addWidget(splitter)

        # Add "LoS" button to toolbar
        toolbar = self.app_window.findChild(QToolBar)
        if toolbar:
            observer_cam_button = QToolButton()
            observer_cam_button.setText("LoS")
            observer_cam_button.setToolTip("Set the camera to the observer's normal direction")
            observer_cam_button.clicked.connect(self.set_camera_to_LOS)

            # toolbar.insertWidget(0, observer_cam_button)
            for action in toolbar.actions():
                if action.text() == "Top (-Z)":
                    toolbar.insertWidget(action, observer_cam_button)
                    break

        self.update_foreground()
        self.app_window.setWindowTitle(f"Refrac Corr Tool")

    def update_y_scale(self, value):
        """
        Update the y scale of the plotter.
        """
        self.set_scale(yscale=value)  # Set the y scale
        self.render()  # Redraw the plotter

    def set_camera_to_LOS(self):
        """
        Set the camera to the line-of-sight (LoS) direction.
        """
        self.view_xz()
        self.camera.ParallelProjectionOn()

    # def shift2pxy(self, shiftx, shifty, freq):
    #     """
    #     Calculate refraction correction coefficients from the given shifts and frequency.
    #
    #     :param shiftx: float
    #         Solar X shift in arcsec.
    #     :param shifty: float
    #         Solar Y shift in arcsec.
    #     :param freq: float
    #         Frequency of the image.
    #     :return: tuple
    #         Computed refraction correction coefficients (px, py).
    #     """
    #     freqhz = np.array(freq) * 1e6
    #     px = shiftx * freqhz ** 2
    #     py = shifty * freqhz ** 2
    #     return px, py
    #
    # def pxy2shifts(self, px, py, freqs):
    #     """
    #     Convert refraction correction coefficients to shifts.
    #
    #     :param px: float
    #         Refraction correction coefficient in x direction.
    #     :param py: float
    #         Refraction correction coefficient in y direction.
    #     :param freqs: list
    #         List of frequencies.
    #     :return: tuple
    #         Solar X and Y shifts (shifts_x, shifts_y).
    #     """
    #     freqshz = np.array(freqs) * 1e6
    #     shifts_x = px[0] * 1 / freqshz ** 2
    #     shifts_y = py[0] * 1 / freqshz ** 2
    #     return shifts_x, shifts_y
    #
    # def correct_images(self):
    #     """
    #     Correct image shifts based on refraction parameters.
    #
    #     :return: np.ndarray
    #         Corrected images.
    #     """
    #     nf, ny, nx = self.image_series.shape
    #     corrected_images = np.zeros((nf, ny, nx))
    #     for idx in range(nf):
    #         shift_x = self.px[0] * 1 / self.cfreqs[idx] ** 2 + self.px[1]
    #         shift_y = self.py[0] * 1 / self.cfreqs[idx] ** 2 + self.py[1]
    #         corrected_images[idx] = np.roll(self.image_series[idx], shift=(int(shift_x), int(shift_y)), axis=(1, 0))
    #     return corrected_images

    def update_foreground(self):
        """
        Update the foreground image data and actors.
        """

        # Save current state before making changes, only if the flag is set
        if self.save_state_flag:
            self.save_rfr_parms_state()

        try:
            self.px0_input.setText(validate_number(self.px0_input.text()))
            self.py0_input.setText(validate_number(self.py0_input.text()))
            self.previous_values['px0'] = self.px0_input.text()
            self.previous_values['py0'] = self.py0_input.text()
        except ValueError:
            self.px0_input.setText(self.previous_values['px0'])
            self.py0_input.setText(self.previous_values['py0'])

        try:
            self.freq_input.setText(
                validate_number_list(self.freq_input.text(), min_value=self.cfreqsmhz[0], max_value=self.cfreqsmhz[-1]))
            self.contourlevels_input.setText(
                validate_number_list(self.contourlevels_input.text(), min_value=0, max_value=100, inclusive=False))
            self.previous_values['freq'] = self.freq_input.text()
            self.previous_values['contourlevels'] = self.contourlevels_input.text()
        except ValueError:
            self.freq_input.setText(self.previous_values['freq'])
            self.contourlevels_input.setText(self.previous_values['contourlevels'])

        self.freq_count_spinbox.blockSignals(True)  # Block signals to prevent triggering callback
        self.freq_count_spinbox.setValue(len(self.freq_input.text().split(',')))
        self.freq_count_spinbox.blockSignals(False)  # Unblock signals

        self.px[0] = -float(self.px0_input.text()) / self.delta_x
        self.py[0] = -float(self.py0_input.text()) / self.delta_y
        self.levels = np.array(list(map(float, self.contourlevels_input.text().split(','))))

        self.freqs = list(map(float, self.freq_input.text().split(',')))
        self.px[1] = 0.0
        self.py[1] = 0.0
        self.shifts_x, self.shifts_y = pxy2shifts(self.px, self.py, self.freqs)

        self.update_frg_contours()
        # print(f"Update time: {time.time() - start} seconds")

    def apply_correction(self):
        """
        Apply the image correction and measure the time taken.

        :return: None
        """

        import time
        start = time.time()
        self.corrected_images = correct_images(self.image_series, self.px, self.py, self.cfreqs)
        print(f"Correction time: {time.time() - start} seconds")

    def update_freqs_from_spinbox(self, value):
        """
        Update frequencies based on spinbox value.

        :param value: int
            Number of frequencies to display.
        """
        min_freq, max_freq = self.cfreqsmhz[0], self.cfreqsmhz[-1]
        self.freqs = np.linspace(min_freq, max_freq, value)
        self.freq_input.setText(', '.join(map(lambda x: f"{x:.3f}", self.freqs)))
        self.validate_and_update_contours()

    def validate_and_update_contours(self):
        """
        Validate the frequency and contour level inputs and update the contours.
        """
        try:
            freqs = validate_number_list(self.freq_input.text(), min_value=self.cfreqsmhz[0],
                                         max_value=self.cfreqsmhz[-1])
            levels = validate_number_list(self.contourlevels_input.text(), min_value=0, max_value=100, inclusive=False)
            self.previous_values['freq'] = self.freq_input.text()
            self.previous_values['contourlevels'] = self.contourlevels_input.text()
        except ValueError:
            self.freq_input.setText(self.previous_values['freq'])
            self.contourlevels_input.setText(self.previous_values['contourlevels'])
            return
        freqs = list(map(float, freqs.split(',')))
        levels = np.array(list(map(float, levels.split(','))))

        if freqs != list(self.freqs):
            self.freqs = freqs
            self.levels = levels
            colors = self.cmap(np.linspace(0, 1, len(self.freqs)))

            # Update existing actors or create new ones
            new_plane_main = []
            existing_freqs = [plane['freq'] for plane in self.plane_main]

            for idx, freq in enumerate(self.freqs):
                if freq in existing_freqs:
                    plane = next(plane for plane in self.plane_main if plane['freq'] == freq)
                    img_data = plane['img_data']
                    if self.rmap_origin is not None:
                        img_data.origin = self.rmap_origin
                else:
                    fidx = np.argmin(np.abs(self.cfreqsmhz - freq))
                    sunpy_map = create_smap(self.image_series[fidx], self.header, freq)
                    plane = self.update_plane_data(sunpy_map, freq, colors[idx])
                new_plane_main.append(plane)

            for plane in self.plane_main:
                if plane['freq'] not in freqs:
                    self.remove_actor(plane['actor_rimg']) if plane['actor_rimg'] is not None else None
                    self.remove_actor(plane['actor_contour']) if plane['actor_contour'] is not None else None
                    self.remove_actor(plane['actor_rdslice']) if plane['actor_rdslice'] is not None else None

            self.plane_main = new_plane_main
            self.update_shifts(self.frg_sphere.GetCenter())
        else:
            self.levels = levels

            self.update_frg_contours()

        self.update_foreground()

    def update_frg_contours(self):
        """
        Update the foreground contours based on refraction parameters.
        """
        if len(self.plane_main) == 0:
            self.init_foreground()
        else:
            colors_frac = np.linspace(0, 1, len(self.plane_main))
            colors = self.cmap(colors_frac)
            for idx, plane in enumerate(self.plane_main):
                plane_origin = plane['origin']
                origin_y = plane_origin[1]
                plane['img_data'].origin = (self.shifts_x[idx] * self.delta_x + plane_origin[0], origin_y,
                                            self.shifts_y[idx] * self.delta_y + plane_origin[2])

                if self.show_rcontours_checkbox.isChecked():
                    self.remove_actor(plane['actor_contour'])
                    conts = plane['img_data'].contour(
                        isosurfaces=self.levels / 100 * np.nanmax(plane['img_data']['scalar']),
                        scalars=plane['img_data']['scalar'])
                    plane['actor_contour'] = self.add_mesh(conts, color=colors[idx], opacity=self.opacity_contours,
                                                           rgb=False,
                                                           render_lines_as_tubes=True,
                                                           line_width=self.contour_line_width,
                                                           show_scalar_bar=False)
                else:
                    self.remove_actor(plane['actor_contour'])
                    plane['actor_contour'] = None

                if self.show_rimg_checkbox.isChecked():
                    self.remove_actor(plane['actor_rimg'])
                    plane['actor_rimg'] = self.add_mesh(plane['img_data'], cmap="viridis", opacity=self.opacity_rimg,
                                                        rgb=False,
                                                        show_scalar_bar=False, pickable=False)
                else:
                    self.remove_actor(plane['actor_rimg'])
                    plane['actor_rimg'] = None

                if self.show_densityslice_checkbox.isChecked():
                    self.remove_actor(plane['actor_rdslice'])
                    img_data = plane['img_data']
                    data_dslice = np.ones_like(img_data['scalar']) * colors_frac[idx] * 255
                    img_data['dslice_data'] = data_dslice
                    opacity_array = norm_to_percent(img_data['scalar'].ravel(order='F'),
                                                    minpercent=np.nanmin(self.levels))
                    plane['actor_rdslice'] = self.add_mesh(img_data, scalars='dslice_data', cmap=self.cmap,
                                                           clim=[0, 255],
                                                           opacity=opacity_array, rgb=False,
                                                           show_scalar_bar=False, pickable=False)
                else:
                    self.remove_actor(plane['actor_rdslice'])
                    plane['actor_rdslice'] = None

    def save_params_to_file(self):
        """
        Save refraction correction parameters to a JSON/CSV file.
        """
        params_filename = define_filename(self.rmap, prefix="refrac_corr_", ext=".csv", get_latest_version=True)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog | QFileDialog.DontConfirmOverwrite 
        filename, _ = QFileDialog.getSaveFileName(self.app_window, "Save", params_filename,
                                                  "CSV Files (*.csv);;JSON Files (*.json)", options=options)
        if filename:
            params = {
                "time": self.rmap.date.to_datetime().strftime("%Y-%m-%dT%H:%M:%S"),
                "px0": float(self.px0_input.text()),
                "py0": float(self.py0_input.text()),
                "px1": float(self.px1_input.text()),
                "py1": float(self.py1_input.text())
            }
            if filename.endswith('.json'):
                with open(filename, 'w') as file:
                    json.dump(params, file, indent=4)
            elif filename.endswith('.csv'):
                # Convert the params dictionary to a DataFrame
                df = pd.DataFrame([params])
                # Check if self.df_params exists, if not create an empty DataFrame
                if self.df_params is None:
                    if os.path.exists(filename):
                        self.df_params = pd.read_csv(filename)
                    else:
                        self.df_params = pd.DataFrame(columns=["time", "px0", "py0", "px1", "py1"])
                self.df_params['time'] = pd.to_datetime(self.df_params['time'])
                df['time'] = pd.to_datetime(df['time'])
                # Merge the df with self.df_params, replacing rows with the same timestamp
                self.df_params = pd.concat([self.df_params[self.df_params['time'] != df['time'].iloc[0]], df])
                # Sort the DataFrame by time
                self.df_params = self.df_params.sort_values(by='time')
                # Save the merged DataFrame to CSV
                self.df_params.to_csv(filename, index=False)
            else:
                print("Invalid file format")
                return

            print(f"Saved refraction corr parameters to {filename}")

    def init_params_plot(self):

        tims_datetime = pd.to_datetime(self.df_params['time'])
        tims = Time(tims_datetime)
        px_values = np.vstack([[self.df_params['px0'].values, self.df_params['px1'].values]]).T
        py_values = np.vstack([[self.df_params['py0'].values, self.df_params['py1'].values]]).T
        # Perform interpolation and plot
        nt = 50
        cadence = (tims[-1].jd - tims[0].jd) / nt
        target_times = Time([tims[0].jd + i * cadence for i in range(nt + 1)], format='jd')

        px_at_targets, py_at_targets = interpolate_rfrcorr_params(
            tims_datetime,
            px_values,
            py_values,
            target_times.to_datetime(),
            methods=(self.method_p0xy.currentText(), self.method_p1xy.currentText())
        )

        tim = self.rmap.date

        px_, py_ = interpolate_rfrcorr_params(
            tims_datetime,
            px_values,
            py_values,
            tim.to_datetime(),
            methods=(self.method_p0xy.currentText(), self.method_p1xy.currentText())
        )
        self.loaded_params = {'px0': px_[0, 0], 'py0': py_[0, 0], 'px1': px_[0, 1], 'py1': py_[0, 1]}

        self.ax_parms[0].cla()
        self.ax_parms[1].cla()
        self.ax_parms[2].cla()
        self.ax_parms[3].cla()

        plot_interpolated_params(
            tims_datetime,
            px_values,
            py_values,
            target_times.to_datetime(),
            px_at_targets,
            py_at_targets,
            save_fig=False,
            fig_parms=self.fig_params,
            ax_parms=self.ax_parms,
            tim=self.rmap.date,
            params=self.loaded_params
        )
        self.sc_px0 = self.ax_parms[0].plot(tim.plot_date, self.loaded_params['px0'], mec='tab:green', mfc='none',
                                            linestyle='',
                                            marker='s', label='current')
        self.sc_py0 = self.ax_parms[1].plot(tim.plot_date, self.loaded_params['py0'], mec='tab:green', mfc='none',
                                            linestyle='',
                                            marker='s', label='current')
        self.sc_px1 = self.ax_parms[2].plot(tim.plot_date, self.loaded_params['px1'], mec='tab:green', mfc='none',
                                            linestyle='',
                                            marker='s', label='current')
        self.sc_py1 = self.ax_parms[3].plot(tim.plot_date, self.loaded_params['py1'], mec='tab:green', mfc='none',
                                            linestyle='',
                                            marker='s', label='current')
        self.canvas_fig_params.draw()
        self.refr_corr_from_params(self.loaded_params)

    def update_params_plot(self):
        if not hasattr(self, 'fig_params') or self.fig_params is None:
            print(f'No figure found. Returning...')
            return
        print(f'Updating plot params data...')
        ax_parms = self.ax_parms
        tim = self.rmap.date
        params = {
            "time": self.rmap.date.to_datetime().strftime("%Y-%m-%dT%H:%M:%S"),
            "px0": float(self.px0_input.text()),
            "py0": float(self.py0_input.text()),
            "px1": float(self.px1_input.text()),
            "py1": float(self.py1_input.text())
        }
        self.sc_px0[0].set_data([tim.plot_date], [params['px0']])
        self.sc_py0[0].set_data([tim.plot_date], [params['py0']])
        self.sc_px1[0].set_data([tim.plot_date], [params['px1']])
        self.sc_py1[0].set_data([tim.plot_date], [params['py1']])
        # print(f'Updated plot data: {params}')

        ax_parms[0].legend(frameon=True, framealpha=0.5)
        ax_parms[1].legend(frameon=True, framealpha=0.5)
        ax_parms[2].legend(frameon=True, framealpha=0.5)
        ax_parms[3].legend(frameon=True, framealpha=0.5)
        self.fig_params.canvas.draw()

    def load_params_from_csv(self, filename):
        # Create a group box for interpolation method selection
        interp_groupbox = QGroupBox("Interpolation Method Selection")
        interp_layout = QVBoxLayout()

        # Create dropdowns for interpolation methods with appropriate labels
        method_p0xy_layout = QHBoxLayout()
        method_p0xy_label = QLabel("σ:")
        self.method_p0xy = QComboBox()
        for method in ['mean', 'fit:linear', 'fit:quadratic', 'interp:linear', 'interp:nearest', 'interp:nearest-up',
                       'interp:zero', 'interp:quadratic', 'interp:cubic', 'interp:previous', 'interp:next']:
            self.method_p0xy.addItem(method)
        method_p0xy_layout.addWidget(method_p0xy_label)
        method_p0xy_layout.addWidget(self.method_p0xy)

        method_p1xy_layout = QHBoxLayout()
        method_p1xy_label = QLabel("Shift:")
        self.method_p1xy = QComboBox()
        for method in ['mean', 'fit:linear', 'fit:quadratic', 'interp:linear', 'interp:nearest', 'interp:nearest-up',
                       'interp:zero', 'interp:quadratic', 'interp:cubic', 'interp:previous', 'interp:next']:
            self.method_p1xy.addItem(method)
        method_p1xy_layout.addWidget(method_p1xy_label)
        method_p1xy_layout.addWidget(self.method_p1xy)

        # Add button to reload the CSV and update the plot
        reload_button = QPushButton('Reload', self)
        reload_button.setToolTip("Reload the CSV file and update the plot")
        reload_button.clicked.connect(lambda: self.load_params_from_csv(filename))

        # Add the layouts to the interpolation group box
        interp_layout.addLayout(method_p0xy_layout)
        interp_layout.addLayout(method_p1xy_layout)
        interp_layout.addWidget(reload_button)
        interp_groupbox.setLayout(interp_layout)

        # Load CSV
        self.df_params = pd.read_csv(filename)

        # Initialize matplotlib figure for plotting
        self.fig_params = Figure(figsize=(4, 7))
        self.canvas_fig_params = FigureCanvas(self.fig_params)
        self.ax_parms = self.fig_params.subplots(4, 1, sharex=True)

        self.toolbar_fig_params = NavigationToolbar(self.canvas_fig_params, self)

        # Initialize params to None
        self.loaded_params = None

        # Connect dropdown changes to update the plot
        self.method_p0xy.currentIndexChanged.connect(self.init_params_plot)
        self.method_p1xy.currentIndexChanged.connect(self.init_params_plot)

        # Update plot initially
        self.init_params_plot()

        # Add the group box and the plot to the layout
        if hasattr(self, 'interp_widget'):
            self.main_ctrl_layout2.removeWidget(self.interp_widget)
            self.interp_widget.deleteLater()

        self.interp_widget = QWidget()
        interp_layout = QVBoxLayout()
        interp_layout.addWidget(interp_groupbox)
        interp_layout.addWidget(self.canvas_fig_params)
        interp_layout.addWidget(self.toolbar_fig_params)
        self.interp_widget.setLayout(interp_layout)
        self.main_ctrl_layout2.addWidget(self.interp_widget)
        self.main_ctrl_layout.parentWidget().setMaximumWidth(850)
        self.main_ctrl_layout.parentWidget().setMinimumWidth(650)

        # Calculate the required width based on the size of the new widget and the existing central layout
        central_layout_width = self.central_layout.sizeHint().width()
        new_width = central_layout_width + self.interp_widget.sizeHint().width() + 20  # Add some padding

        # Adjust the window size to accommodate the new widget
        self.app_window.setMinimumWidth(new_width)
        self.app_window.resize(new_width, self.app_window.height())
        return self.loaded_params

    def load_params_from_file(self):
        """
        Load refraction correction parameters from a JSON file.
        """
        params_filename = define_filename(self.rmap, prefix="refrac_corr_", ext=".csv", get_latest_version=True)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self.app_window, "Load", params_filename,
                                                  "CSV Files (*.csv);;JSON Files (*.json)",
                                                  options=options)
        if filename:
            # try:
            if filename.endswith('.csv'):
                # Load CSV
                params = self.load_params_from_csv(filename)
            elif filename.endswith('.json'):
                # Load JSON
                with open(filename, 'r') as file:
                    params = json.load(file)
                self.refr_corr_from_params(params)

            print(f"Loaded refraction corr parameters from {filename}")
            # except Exception as e:
            #     print(f"Failed to load parameters: {str(e)}")

    def refr_corr_from_params(self, params):
        current_state = self.current_state
        self.save_state_flag = False
        self.px0_input.setText(f"{params['px0']}")
        self.py0_input.setText(f"{params['py0']}")
        self.px1_input.setText(f"{params['px1']}")
        self.py1_input.setText(f"{params['py1']}")
        self.move_frg_sphere()
        self.move_bkg_sphere()
        self.save_state_flag = True
        self.save_rfr_parms_state(current_state)
        return

    def save_rfr_parms_state(self, previous_values=None):
        """
        Save the current state of refraction parameters.
        """
        if previous_values is None:
            previous_values = self.previous_values
        state = {k: str(v) for k, v in previous_values.items()}
        # if state in self.undo_stack:
        #     print("this state already exists in the stack, not saving again")
        #     return
        self.undo_stack.append(state)

        self.redo_stack.clear()  # Clear the redo stack whenever a new change is made

        # Limit the size of undo_stack to 50 elements
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def load_rfr_parms_state(self, state):
        """
        Load a saved state of refraction parameters.

        :param state: dict
            Saved state of refraction parameters.
        """
        self.save_state_flag = False  # Temporarily disable state saving to avoid loop
        self.px0_input.setText(state["px0"])
        self.py0_input.setText(state["py0"])
        self.px1_input.setText(state["px1"])
        self.py1_input.setText(state["py1"])
        self.freq_input.setText(state["freq"])
        self.contourlevels_input.setText(state["contourlevels"])
        self.move_frg_sphere()
        self.move_bkg_sphere()
        self.save_state_flag = True  # Re-enable state saving

    @property
    def current_state(self):
        current_state = {
            "px0": self.px0_input.text(),
            "py0": self.py0_input.text(),
            "px1": self.px1_input.text(),
            "py1": self.py1_input.text(),
            "freq": self.freq_input.text(),
            "contourlevels": self.contourlevels_input.text()
        }
        return current_state

    def undo_rfr_parms(self):
        """
        Undo the last change to refraction parameters.
        """

        if self.undo_stack:
            current_state = self.current_state
            self.redo_stack.append(current_state)
            last_state = self.undo_stack.pop()
            # print("set params to :", last_state)
            self.load_rfr_parms_state(last_state)

    def redo_rfr_parms(self):
        """
        Redo the last undone change to refraction parameters.
        """
        if self.redo_stack:
            current_state = self.current_state
            self.undo_stack.append(current_state)
            next_state = self.redo_stack.pop()
            self.load_rfr_parms_state(next_state)

    def update_contour_opacity_from_slider(self, value):
        """
        Update contour opacity based on slider value.

        :param value: int
            Slider value for opacity.
        """
        self.opacity_contours = value / 100.0
        self.opacity_rimg = self.opacity_contours
        self.apply_opacity_changes()

    def update_contour_opacity_from_input(self):
        """
        Update contour opacity based on input value.
        """
        value = int(self.opacity_control.input_box.text())
        self.opacity_control.slider.setValue(value)
        self.update_contour_opacity_from_slider(value)

    def apply_opacity_changes(self):
        """
        Apply opacity changes to the contours and images.
        """

        if self.show_rcontours_checkbox.isChecked():
            for plane in self.plane_main:
                if plane['actor_contour'] is not None:
                    plane['actor_contour'].GetProperty().SetOpacity(self.opacity_contours)
        if self.show_rimg_checkbox.isChecked():
            for plane in self.plane_main:
                if plane['actor_rimg'] is not None:
                    plane['actor_rimg'].GetProperty().SetOpacity(self.opacity_rimg)
        # if self.show_densityslice_checkbox.isChecked():
        #     colors = self.cmap(np.linspace(0, 1, len(self.plane_main)))
        #     for idx, plane in enumerate(self.plane_main):
        #         if plane['actor_rdslice'] is not None:
        #             img_data = plane['img_data']
        #             c = colors[idx]
        #             img_data['dslice_data'] = data1d_to_rgba(copy(img_data['scalar']), c, alpha=self.opacity_rimg,
        #                                                        minpercent=np.nanmin(self.levels))
        self.render()

    def update_contour_lw_from_slider(self, value):
        """
        Update contour line width based on slider value.

        :param value: float
            Slider value for line width.
        """
        self.contour_line_width = float(value)
        self.apply_line_width_changes()

    def update_contour_lw_from_input(self):
        """
        Update contour line width based on input value.
        """
        value = float(self.contour_lw_control.input_box.text())
        self.contour_lw_control.slider.setValue(int(value))
        self.update_contour_lw_from_slider(self.contour_lw_control.slider.value())

    def apply_line_width_changes(self):
        """
        Apply line width changes to the contours.
        """
        for plane in self.plane_main:
            if plane['actor_contour'] is not None:
                plane['actor_contour'].GetProperty().SetLineWidth(self.contour_line_width)
        self.render()

    def reset_freqs_to_default(self):
        """
        Reset the frequencies to default values.
        """
        self.freq_input.setText(self.freqs_default)
        self.validate_and_update_contours()


def main():
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Run the Image Correction Application for OVRO-LWA data.')
    parser.add_argument('--lwafile', type=str, help='Path to the LWA HDF5/FITS file.')
    parser.add_argument('--background_map', type=str, nargs='+', help='List of background map files.')
    parser.add_argument('--freqs', type=str, help='Comma-separated list of frequencies to display.')
    parser.add_argument('--trajectory_file', type=str, default='', help='Path to the trajectory file')

    # Parse the arguments
    args = parser.parse_args()

    # Start the Qt application
    app = QApplication([])
    # Initialize the Image Correction Application with the provided arguments
    imgcorr = ImageCorrectionApp(lwafile=args.lwafile, background_map=args.background_map, freqs_default=args.freqs,
                                 trajectory_file=args.trajectory_file)
    imgcorr.show()
    # Execute the application and exit
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
