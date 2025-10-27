# A script to facilitate the use of the DRAGONS software for reducing Gemini data.
# Script usage is 'python main.py /path/to/gemini_data --science_exptime=<exptime>'
# 'gemini_data' should be the directory containing the raw science files for the program, and also the calibration files as suggested by the Gemini archive.
# NB: There are some calibration files with data labels of the form 'G[N,S]-CALYYYYMMDD-#-###-G-BIAS', with filenames prefixed with a 'g'.  I do not know what these files are for, they break the current script, and data reductions are possible without them, so for the time being do not include these files in the 'gemini_data' directory (all other calibration files should be there).

import argparse
import bz2
import glob
import os
import os.path as pa
import tarfile

import astrodata
import astropy.units as u
import gemini_instruments
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system import cal_service
from recipe_system.reduction.coreReduce import Reduce

import configure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to data; will attempt to interpret as .tar archive or directory.",
    )
    parser.add_argument(
        "--science_exptime",
        type=float,
        default=None,
        help="Exposure time of science frames (used to select science frames; default is None (unused)).",
    )
    return parser.parse_args()


def unpack_directory(data_path):
    # Extract tar file
    if data_path.endswith(".tar"):
        print(f"Extracting {data_path}...")
        data_dir = pa.splitext(data_path)[0]
        # Make directory for extracted files
        if not pa.exists(data_dir):
            os.makedirs(data_dir)
        # Extract to directory
        with tarfile.open(data_path, "r") as tar:
            tar.extractall(path=data_dir)
        data_dir = pa.splitext(data_path)[0]
    # Use directory directly
    elif pa.isdir(data_path):
        print(f"Using directory {data_path}...")
        data_dir = data_path
    else:
        raise ValueError(
            "data_path must be a directory or a .tar file containing data."
        )
    # Check if data directory exists
    if not pa.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    return data_dir


def bunzip_files(data_dir):
    print(f"Unzipping .bz2 files in the directory {data_dir}...")
    # Find all .bz2 files in the directory
    bz2_files = glob.glob(f"{data_dir}/*.bz2")
    if not bz2_files:
        print("No .bz2 files found in the directory.")
        return

    # Unzip files
    for bz2_file in bz2_files:
        with bz2.open(bz2_file, "rb") as f_in:
            file_name = bz2_file.strip(".bz2")  # Remove the .bz2 extension
            if not pa.exists(file_name):
                print(
                    f"Unzipping {pa.basename(bz2_file)} to {pa.basename(file_name)}..."
                )
                with open(file_name, "wb") as f_out:
                    f_out.write(f_in.read())


def categorize_files(data_dir, science_exptime=None):
    # Get all files
    files = {
        "all": np.sort(glob.glob(f"{data_dir}/*.fits")),
    }

    # Print file infos
    print("Preliminary file info (name, exposure time, tags):")
    for f in files["all"]:
        ad = astrodata.open(f)
        print("\t", pa.basename(f), f"{ad.exposure_time()}s", ad.tags)

    # Bias
    files["bias"] = dataselect.select_data(files["all"], ["BIAS"])
    # Bias centspec
    files["bias_centspec"] = dataselect.select_data(
        files["bias"],
        expression=dataselect.expr_parser("detector_roi_setting=='Central Spectrum'"),
    )
    # Bias fullframe
    files["bias_fullframe"] = dataselect.select_data(
        files["bias"],
        expression=dataselect.expr_parser("detector_roi_setting=='Full Frame'"),
    )
    # Darks
    files["dark"] = dataselect.select_data(files["all"], ["DARK"])
    # Flats
    files["flat"] = dataselect.select_data(files["all"], ["FLAT"])
    # Arcs
    files["arc"] = dataselect.select_data(files["all"], ["ARC"])
    # Standards
    files["standard"] = dataselect.select_data(files["all"], ["STANDARD"])
    # Science
    if science_exptime is not None:
        files["science"] = dataselect.select_data(
            files["all"],
            # xtags=["CAL", "AQUISITION"],
            expression=dataselect.expr_parser(f"exposure_time=={science_exptime}"),
        )
    else:
        raise NotImplementedError(
            "Science frame selection without exposure time is not implemented; please provide --science_exptime as a command line argument."
        )
    # BPMs
    files["bpm"] = dataselect.select_data(files["all"], ["BPM"])

    # Print categorized files
    print("Categorized files:")
    for k, i in files.items():
        print(f"\t{k} ({len(i)} files):")
        for f in i:
            print("\t\t", pa.basename(f))

    return files


def run_reduction(data_dir, science_exptime=None):
    # Connect to local db
    caldb = cal_service.set_local_database()

    # Setup logging
    logutils.config(file_name=f"{data_dir}/reduction.log")

    # Get sorted files
    files = categorize_files(data_dir, science_exptime=science_exptime)

    # Make reduction directory; change to there
    reduce_dir = f"{data_dir}/reduction"
    if not pa.exists(reduce_dir):
        print(f"Creating reduction directory {reduce_dir}...")
        os.makedirs(reduce_dir)
    os.chdir(reduce_dir)

    ##############################
    ###    Bad Pixel Masks     ###
    ##############################

    # Add static BPMs to local calibration database
    for bpm in files["bpm"]:
        caldb.add_cal(bpm)

    # # Create user bpm
    # reduce_bpm = Reduce()
    # reduce_bpm.files.extend(files["flat"])
    # reduce_bpm.files.extend(files["dark"])
    # reduce_bpm.recipename = "makeProcessedBPM"
    # reduce_bpm.runr()
    # user_bpm = reduce_bpm.output_filename[0]
    # print(user_bpm)

    ##############################
    ###          Bias          ###
    ##############################

    reduce_bias = Reduce()
    reduce_bias.files.extend(files["bias_centspec"])
    # reduce_bias.uparms = [("addDQ:user_bpm", user_bpm)]
    reduce_bias.runr()

    if files["bias_fullframe"]:
        reduce_bias = Reduce()
        reduce_bias.files.extend(files["bias_fullframe"])
        # reduce_bias.uparms = [("addDQ:user_bpm", user_bpm)]
        reduce_bias.runr()

    ##############################
    ###      Master Flat       ###
    ##############################

    reduce_flat = Reduce()
    reduce_flat.files.extend(files["flat"])
    # reduce_flat.uparms = [("addDQ:user_bpm", user_bpm)]
    reduce_flat.runr()

    ##############################
    ###          Arcs          ###
    ##############################

    reduce_arc = Reduce()
    reduce_arc.files.extend(files["arc"])
    # reduce_arc.uparms = [("addDQ:user_bpm", user_bpm)]
    # reduce_arc.uparms = [("interactive", True)]
    reduce_arc.runr()

    ##############################
    ###        Standard        ###
    ##############################

    reduce_standard = Reduce()
    reduce_standard.files.extend(files["standard"])
    # reduce_standard.uparms = [("addDQ:user_bpm", user_bpm)]
    # reduce_standard.uparms.update(dict([("darkCorrect:do_cal", "skip")]))
    reduce_standard.uparms.update({"darkCorrect:do_cal": "skip"})
    reduce_standard.runr()

    ##############################
    ###        Science         ###
    ##############################

    reduce_science = Reduce()
    reduce_science.files.extend(files["science"])
    # reduce_science.uparms = [('addDQ:user_bpm', user_bpm)]
    # reduce_science.uparms.update(dict([("skyCorrect:scale_sky", False)]))
    reduce_science.uparms.update({"skyCorrect:scale_sky": False})
    reduce_science.runr()

    return reduce_dir


def quicklook(reduce_dir):
    # Get file
    files_1D = glob.glob(f"{reduce_dir}/*_1D.fits")
    # Iterate over files
    for f in files_1D:
        with fits.open(f) as hdul:
            print(hdul.info())
            i_sci = 1
            while i_sci < len(hdul) - 2:
                # Set indices
                i_var = i_sci + 1
                i_dq = i_sci + 2
                # Get data
                print(hdul[i_sci].header)
                sci = hdul[i_sci].data
                var = hdul[i_var].data
                dq = hdul[i_dq].data
                # Get wavelengths
                wcs = WCS(hdul[i_sci].header)
                lam = wcs.pixel_to_world(np.arange(sci.shape[0])).to(u.AA)
                # Mask
                dq_mask = dq == 0
                lam_masked = lam[dq_mask]
                sci_masked = sci[dq_mask]
                var_masked = var[dq_mask]
                # Normalize data
                sci_normed = sci_masked  # / np.nanmedian(sci_masked)
                std_normed = np.sqrt(var_masked)  # / np.nanmedian(sci_masked)
                # Plot
                plt.figure(figsize=(10, 4))
                idxstr = f"[{i_sci}-{i_dq}]"
                label = f"{pa.basename(f)}{idxstr}"
                sc = SkyCoord(
                    hdul[i_sci].header["XTRACTRA"],
                    hdul[i_sci].header["XTRACTDE"],
                    unit=u.deg,
                )
                rdstr = f"({sc.to_string('hmsdms', precision=1)})"
                x = lam_masked.value
                y = sci_normed
                ylo = sci_normed - std_normed
                yhi = sci_normed + std_normed
                plt.fill_between(
                    x,
                    ylo,
                    yhi,
                    color="xkcd:ocean",
                    alpha=0.5,
                )
                plt.plot(
                    x,
                    y,
                    color="k",
                )
                plt.xlabel("Wavelength (AA)")
                plt.ylabel("Flux (normalized)")
                plt.title(
                    f"{label} (Spectrum {int(np.ceil(i_sci / 3))}/{len(hdul) // 3 - 1}) {rdstr}"
                )
                # Most of the time, the flux values go crazy at the blue end, so we set the limits based on the middle portion
                # Trim
                ilo = int(len(x) * 0.15)
                ihi = int(len(x) * 1.0)
                # Get min and max of flux
                y_min = np.nanmin(y[ilo:ihi])
                y_max = np.nanmax(y[ilo:ihi])
                # Set limits to slightly below and above the min and max
                buffer = 0.1 * (y_max - y_min)
                plt.ylim(y_min - buffer, y_max + buffer)
                # Clean + save
                plt.tight_layout()
                plt.savefig(f.replace(".fits", f"{idxstr}.png"))
                plt.close()
                # Iterate
                i_sci += 3


def main():
    # Get command line arguments
    args = parse_args()

    # Configure
    configure.main()

    # Setup directory
    data_dir = unpack_directory(args.data_path)

    # Unzip files
    bunzip_files(data_dir)

    # Run reduction
    reduce_dir = run_reduction(data_dir, science_exptime=args.science_exptime)

    # Make quicklook
    quicklook(reduce_dir)

    return reduce_dir

if __name__ == "__main__":
    main()
