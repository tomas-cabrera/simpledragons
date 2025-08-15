import os
import os.path as pa
import shutil

from recipe_system import cal_service

# Get path to project
PROJPATH = os.path.dirname(os.path.abspath(__file__))
CACHEPATH = f"{PROJPATH}/.dragons"


def configure(
    browser="firefox",
    cachepath=CACHEPATH,
    force=False,
):
    print("Configuring DRAGONS...")

    # dragonsrc
    srcpath = f"{CACHEPATH}/dragonsrc"
    os.environ["DRAGONSRC"] = srcpath
    dbpath = f"{CACHEPATH}/dragons.db"
    if not pa.exists(srcpath) or force:
        print("Initializing dragonsrc...")
        # Make directory
        if not pa.exists(pa.dirname(srcpath)):
            os.makedirs(pa.dirname(srcpath))
        # Make dragonsrc file
        with open(srcpath, "w") as f:
            f.write("[interactive]\n")
            f.write(f"browser = {browser}\n")
            f.write("\n")
            f.write("[calibs]\n")
            f.write(f"databases = {dbpath} get store\n")

    # ds9 buffers
    imtoolrc_path = f"{PROJPATH}/.imtoolrc"
    if not pa.exists(imtoolrc_path) or force:
        print(
            "The DRAGONS documentation requires the ds9 buffer configuration to be changed.",
            "This defines/updates the IMTOOLRC environment variable in your shell environment.",
            "More details: https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/v4.0.0/install.html#configure-dragons",
            f"Current IMTOOLRC: {os.environ.get('IMTOOLRC', '<undefined>')}",
            f"Proposed IMTOOLRC: {imtoolrc_path}",
            sep="\n",
        )
        do_update_imtoolrc = input("Update IMTOOLRC? (y/n) ")
        if do_update_imtoolrc.lower() == "y":
            # Copy imtoolrc file
            shutil.copyfile(
                f"{os.environ['CONDA_PREFIX']}/lib/python3.12/site-packages/gempy/numdisplay/imtoolrc",
                imtoolrc_path,
            )
            # Set environment variable
            os.environ["IMTOOLRC"] = imtoolrc_path
            print(
                "Updated IMTOOLRC.",
                "If you use this variable elsewhere, you may need to redefine it after running this script.",
                sep="\n",
            )
        else:
            print("IMTOOLRC not updated; DRAGONS may not behave as expected.")

    # Initialize db
    caldb = cal_service.set_local_database()
    if not pa.exists(dbpath):
        print("Initializing calibration database...")
        caldb.init()

    # Update .bash_profile
    print(
        "DRAGONS configuration complete.",
        "It is recommended to update your .bash_profile;",
        "to do so add the following lines to your ~/.bash_profile:",
        f"\texport IMTOOLRC={imtoolrc_path}",
        f"\texport DRAGONSRC={srcpath}",
        "\tulimit -n 2048",
        sep="\n",
    )


def test_installation():
    import astrodata
    import gemini_instruments
    from recipe_system.reduction.coreReduce import Reduce

    print("Test packages imported successfully.")


def main():
    configure()
    test_installation()


if __name__ == "__main__":
    main()
