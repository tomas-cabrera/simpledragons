# SIMPLEDRAGONS

Some code to make it easy to use the NOIRLab DRAGONS package to reduce data.
Originally written for GMOS spectra; will be updated with additional routines as appropriate.

Steps:
1. Set up the environment as normal for conda (`conda create --file environment.yaml`).
2. Download the relevant science and calibration data (usually two separate .tar archives).
3. Unpack the archives into a single directory (bunzipping the files optional, should be handled by this code if needed).
4. Run the script: `python main.py --science_exptimes <time> <path/to/data/dir>`.

Notes:
- There isn't a clear way to identify the science exposures yet.  A better way may be to keep them in a separate directory as this matches up with the archive downloads, but for now you just have to set the exptime in the command line (this clearly won't work if some files have the same exptime as the science images).
- Written for DRAGONS v4.0.0, as hinted in `environment.yaml`.