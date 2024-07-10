
# Image Correction Pipeline

This project provides a  solution for correcting the refraction effects in OVROLWA images. It is designed to work with a range of frequencies and supports various image file types.

## Features

- **Frequency Support**: Handles a wide range of frequencies specified in MHz.
- **Normalization**: Supports both logarithmic and standard normalization for image processing.
- **File Type Support**: Compatible with multiple image file types for input and output.
- **Spectral Mode**: Allows for different modes of spectral analysis.
- **Correction Parameters**: Utilizes radio frequency interference correction parameters for image correction.
- **Contour Drawing**: Option to draw contours on images for enhanced analysis.
- **Dual Panel Display**: Supports displaying images in a dual-panel mode for comparative analysis.
- **Version Control**: Automatically fetches the latest version of correction parameters.
- **Flexible Time Difference Tolerance**: Configurable tolerance for time difference in image timestamps.

## Installation

To set up this project, ensure you have Python and pip installed on your system. Then, run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the image correction pipeline, run the main script with the necessary arguments. Here is an example command:

```bash
python img_corr_pipeline.py --mode <mode> --freq <frequency> --filetype <file_type> --specmode <spectral_mode> --norm <normalization> --docorr <do_correction> --workdir <working_directory> --alpha <alpha_value> --minpercent <minimum_percentage> --draw_contours <draw_contours> --fov <field_of_view> --dual_panel <dual_panel> --get_latest_version <get_latest_version> --timediff_tol <time_difference_tolerance> --rfrcor_parm_files <rfrcor_parm_files_path> --interp_method <interpolation_method> --overwrite <overwrite>
```

Replace the placeholders with your specific configurations.

## Contributing

Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License
