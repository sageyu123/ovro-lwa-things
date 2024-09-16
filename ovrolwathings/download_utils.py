import configparser
import os
import re
from datetime import datetime, timedelta
from getpass import getpass

import astropy.units as u
import numpy as np
import paramiko
from astropy.time import Time


def get_file_list(ssh, remote_dir):
    """Get the list of files from the remote directory."""
    stdin, stdout, stderr = ssh.exec_command(f"ls {remote_dir}")
    files = stdout.read().decode().split()
    return files


def find_nearest_file(files, timestamp, time_format, diff_tol, specmode='mfs', filetype='hdf'):
    """Find the nearest file for each timestamp within a time tolerance."""
    if filetype == 'hdf':
        file_pattern = re.compile(rf".*\.lev\d+_{specmode}_\d+s\.(\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z)\.(image|image_I)\.hdf")
    elif filetype == 'fits':
        file_pattern = re.compile(rf".*\.lev\d+_{specmode}_\d+s\.(\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z)\.(image|image_I)\.fits")
    else:
        print(f"Invalid filetype: {filetype}. Must be 'hdf' or 'fits'. Defaulting to 'hdf'.")
        file_pattern = re.compile(rf".*\.lev\d+_{specmode}_\d+s\.(\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z)\.(image|image_I)\.hdf")

    # print(file_pattern)
    # print(files)
    # Extract file times
    file_times = []
    valid_files = []
    for file in files:
        match = file_pattern.match(file)
        if match:
            file_time_str = match.group(1)
            file_time = datetime.strptime(file_time_str, time_format)
            file_times.append(file_time)
            valid_files.append(file)

    if len(file_times) == 0:
        return None
    # Convert file times to JD
    file_times_jd = Time(file_times).jd

    # Convert timestamp to JD
    timestamp_jd = Time([timestamp]).jd

    # Find the nearest file within the time tolerance
    nearest_file = None

    time_diff = np.abs(timestamp_jd - file_times_jd)
    idx = np.nanargmin(time_diff)
    if time_diff[idx] <= diff_tol:
        nearest_file = valid_files[idx]

    return nearest_file


def download_files(user, hostname, files, remote_dirs, local_base_dirs, specmode='mfs', filetype='hdf'):
    """Download files using scp and maintaining directory structure."""
    downloaded_files = []
    for idx, file in enumerate(files):
        if file is None:
            continue

        # Extract date from filename
        if filetype == 'hdf':
            file_pattern = re.compile(rf".*\.lev\d+_{specmode}_\d+s\.(\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z)\.(image|image_I)\.hdf")
        elif filetype == 'fits':
            file_pattern = re.compile(rf".*\.lev\d+_{specmode}_\d+s\.(\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z)\.(image|image_I)\.fits")
        else:
            print(f"Invalid filetype: {filetype}. Must be 'hdf' or 'fits'. Defaulting to 'hdf'.")
            file_pattern = re.compile(rf".*\.lev\d+_{specmode}_\d+s\.(\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z)\.(image|image_I)\.hdf")

        date_str = file_pattern.match(file).group(1).split("T")[0]
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Create remote and local directories
        remote_dir = remote_dirs[idx].format(year=date.year, month=date.month, day=date.day)
        local_dir = os.path.join(local_base_dirs[idx], str(date.year), f"{date.month:02d}", f"{date.day:02d}")
        os.makedirs(local_dir, exist_ok=True)

        # Check if the file already exists locally
        local_file_path = os.path.join(local_dir, file)

        if not os.path.exists(local_file_path):
            # Download the file
            os.system(f"scp {user}@{hostname}:{remote_dir}/{file} {local_file_path}")
        else:
            print(f"File {local_file_path} already exists locally. Skipping download.")
        downloaded_files.append(local_file_path)
    return downloaded_files


def get_local_file_list(local_base_dir, specmode, filetype='hdf'):
    """Get the list of local files."""
    if filetype == 'hdf':
        filenamesuffixes = ['.image_I.hdf', '.image.hdf']
    elif filetype == 'fits':
        filenamesuffixes = ['.image_I.fits', '.image.fits']
    else:
        filenamesuffixes = ['.image_I.hdf', '.image.hdf']

    local_files = []
    for root, _, files in os.walk(local_base_dir):
        for file in files:
            if specmode in file and any(file.endswith(suffix) for suffix in filenamesuffixes):
                local_files.append(os.path.relpath(os.path.join(root, file), local_base_dir))
    return local_files


def setup_ssh_connection(user, hostname, identityfile):
    """Setup the SSH connection."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        rsa_key = paramiko.RSAKey(filename=identityfile)
    except paramiko.PasswordRequiredException:
        password = getpass("Enter passphrase for private key: ")
        rsa_key = paramiko.RSAKey(filename=identityfile, password=password)
    try:
        ssh.connect(hostname=hostname, username=user, pkey=rsa_key)
    except paramiko.AuthenticationException as e:
        print(f"Authentication failed: {e}")
        exit(1)
    except paramiko.SSHException as e:
        print(f"SSH connection failed: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
    return ssh


# Main function to download OVRO-LWA data
def download_ovrolwa(starttime=None, endtime=None, cadence=None, timestamps=None,
                     data_dir='./',
                     mode='auto', level='lev1',
                     filetype='hdf', timediff_tol=None, specmode='mfs',
                     config='ssh-to-data.private.config'):
    '''
    Download OVRO-LWA data from the OVSA server for a given time range or list of timestamps.
    :param starttime:
    :param endtime:
    :param cadence:
    :param timestamps:
    :param mode:
    :param level:
    :param timediff_tol: time tolerance in seconds
    :param specmode: 'fch' or 'mfs'
    :param config: directory to file containing SSH config
    :return:
    '''

    # Read SSH config
    # check if the file exists
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file {config} not found.")

    sshconfig = configparser.ConfigParser()
    sshconfig.read(config)
    ssh_host = sshconfig['SSH']['host']
    user = sshconfig['SSH']['user']
    identityfile = os.path.expanduser(sshconfig['SSH']['identityfile'])

    # Define other constants
    remote_dir_template = "/nas6/ovro-lwa-data/{filetype}/{mode}/{level}/{year}/{month:02d}/{day:02d}"
    local_base_dir_template = f"{data_dir}/ovro-lwa-data/{filetype}/{{mode}}/{level}/"
    time_format = "%Y-%m-%dT%H%M%SZ"
    # Set default diff_tol based on mode if not provided
    if timediff_tol is None:
        slow_diff_tol = 60 * u.second
        fast_diff_tol = 10 * u.second
    else:
        if not isinstance(timediff_tol, u.Quantity):
            slow_diff_tol = fast_diff_tol = timediff_tol * u.second
        else:
            slow_diff_tol = fast_diff_tol = timediff_tol

    slow_diff_tol = slow_diff_tol.to(u.day).value
    fast_diff_tol = fast_diff_tol.to(u.day).value

    if timestamps is None:
        if starttime is None or endtime is None or cadence is None:
            raise ValueError("For time range input, starttime, endtime, and cadence must be provided")
        timestamps = [starttime + i * cadence for i in range(int((endtime - starttime) / cadence) + 1)]
    else:
        if not isinstance(timestamps, list) or not all(isinstance(ts, datetime) for ts in timestamps):
            raise ValueError("timestamps must be a list of datetime objects")
        starttime = timestamps[0]
        endtime = timestamps[-1]

    local_base_dirs = {
        'slow': local_base_dir_template.format(mode='slow'),
        'fast': local_base_dir_template.format(mode='fast')
    }
    diff_tols = {
        'slow': slow_diff_tol,
        'fast': fast_diff_tol
    }
    modes_to_check = ['slow', 'fast'] if mode == 'auto' else [mode]

    # Find nearest local files
    nearest_local_files = []
    for timestamp in timestamps:
        nearest_file = None
        for mode_check in modes_to_check:
            local_files = get_local_file_list(local_base_dirs[mode_check], specmode, filetype)
            nearest_file = find_nearest_file(local_files, timestamp, time_format, diff_tols[mode_check], specmode,
                                             filetype)
            if nearest_file is not None:
                nearest_file = os.path.join(local_base_dirs[mode_check], nearest_file)
                break
        nearest_local_files.append(nearest_file)

    # print(f'Nearest local files: {nearest_local_files}, timestamps: {timestamps}')
    # If all files are found locally, return them
    local_files_found = list(filter(None, nearest_local_files))
    print(f'{len(local_files_found)} out of {len(timestamps)} files at the provided timestamps found locally.')
    if len(local_files_found) == len(timestamps):
        print("All files found locally. No need to download.")
        return nearest_local_files

    print('Trying to download missing files...')
    ssh = setup_ssh_connection(user, ssh_host, identityfile)

    nearest_remote_files = []
    remote_dirs = []
    local_dirs = []
    date_ref = timestamps[0].date()
    remote_dir_options = {}
    remote_files_options = {}
    for mode_check in modes_to_check:
        remote_dir = remote_dir_template.format(mode=mode_check, level=level, filetype=filetype, year=date_ref.year,
                                                month=date_ref.month, day=date_ref.day)
        remote_dir_options[mode_check] = remote_dir
        remote_files_options[mode_check] = get_file_list(ssh, remote_dir)

    # print(f"Remote directories: {remote_dir_options}")
    # print(f"Remote files: {remote_files_options}")
    for i, timestamp in enumerate(timestamps):
        if nearest_local_files[i] is None:
            if timestamp.date() != date_ref:
                for mode_check in modes_to_check:
                    remote_dir = remote_dir_template.format(mode=mode_check, level=level, filetype=filetype,
                                                            year=date_ref.year,
                                                            month=date_ref.month, day=date_ref.day)
                    remote_dir_options[mode_check] = remote_dir
                    remote_files_options[mode_check] = get_file_list(ssh, remote_dir)
            nearest_file = None
            # import IPython;
            # IPython.embed()
            for mode_check in modes_to_check:
                nearest_file = find_nearest_file(remote_files_options[mode_check], timestamp, time_format,
                                                 diff_tols[mode_check], specmode, filetype)
                if nearest_file is not None:
                    nearest_remote_files.append(nearest_file)
                    local_dirs.append(local_base_dirs[mode_check])
                    remote_dirs.append(remote_dir_options[mode_check])
                    break
            if nearest_file is None:  # If no file is found within the time tolerance
                nearest_remote_files.append(None)
                local_dirs.append(None)
                remote_dirs.append(None)
        else:
            nearest_remote_files.append(None)
            local_dirs.append(None)
            remote_dirs.append(None)

    # Close SSH connection
    ssh.close()

    remote_files_found = list(filter(None, nearest_remote_files))
    if len(remote_files_found)==0:
        print("None of the missing files found on the server. No files downloaded.")
    # Download the missing files
    downloaded_files = download_files(user, ssh_host, nearest_remote_files,
                                      remote_dirs,
                                      local_dirs,
                                      specmode,
                                      filetype)

    returnfiles = []
    for idx, file in enumerate(nearest_local_files):
        if file is not None:
            returnfiles.append(file)
        else:
            if idx < len(downloaded_files):
                returnfiles.append(downloaded_files[idx])
            else:
                returnfiles.append(None)
    return returnfiles


if __name__ == "__main__":
    # from utils import download_ovrolwa
    # from datetime import datetime, timedelta

    start_time = datetime(2024, 6, 1, 0, 0, 0)
    end_time = datetime(2024, 6, 1, 0, 1, 0)
    cadence = timedelta(seconds=10)
    mode = 'auto'  # 'fast', 'slow', or 'auto'
    specmode = 'mfs'  # or 'fch'
    timestamps = [start_time + i * cadence for i in range(int((end_time - start_time) / cadence) + 1)]

    downloaded_files = download_ovrolwa(timestamps=timestamps, mode=mode, specmode=specmode)
    print("Downloaded files:", downloaded_files)
