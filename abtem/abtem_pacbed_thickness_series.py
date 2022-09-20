"""
This script generates PACBED images based on an input .cif file, and automatically creates
a thickness series of images.  The settings which may commonly be adjusted are collected in
the settings dictionary just after the imports.

The basic logic of the script is:
    1.  Read the .cif file
    2.  Initialize a pixelated detector to caputre the PACBEDs
    3.  Generate a full-thickness model from the .cif file
    4.  Setup other needed objects (probe, gridscan, frozen phonon configs, etc)
    5.  Generate a potential object for each frozen phonon configuration
    6.  Slice the potential into chunks (of "thiickness_step") along propogation direction
    7.  Propogate the probe through that chunk for each position in the gridscan
    8.  Detect the probes and store those measurements into a list
    9.  Sum together the measurement from each probe position to get PACBEDs for each thickness
    10. Export the PACBEDs as a .tif stack
"""


# %% IMPORTS
# ASE Imports
import ase.io as aio
import ase.build as abuild
# from ase.visualize import view as aview
# Useful for checking generated structure
# AbTEM Imports
from abtem.potentials import Potential
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.structures import orthogonalize_cell
from abtem.detect import PixelatedDetector
from abtem.temperature import FrozenPhonons
# Other Imports
import cupy
import os
from timeit import default_timer as timer
import warnings
from tqdm import tqdm
import tifffile
from math import sin, cos, radians, degrees

# %% SETTINGS
prms = {"seed":            42,              # Pseudo-random seed
        "device":          "gpu",           # Set computing device, must be "gpu" or "cpu"
        "gpu_num":         0,               # If "device" == "gpu", which GPU to use (0 or 1)
        # STRUCTURE FILE LOCATION
        "path":            r"E:\Users\Charles\BTO PACBED\abtem",
        "filename":        "BaTiO3_mp-2998_conventional_standard.cif",
        # POTENTIAL SETTINGS
        "sampling":        0.2,             # Sampling of potential (A)
        "tiling":          30,              # Number of times to tile projected cell
        "thickness":       200,             # Total model thickness (A)
        "thickness_step":  10,              # PACBED export thickness steps (A)
        "slice_thickness": 2,               # Thickness per simulation slice (A)
        "zas":             [(0, 0, 1)],     # Zone axes to model
        "fp_configs":      10,              # Number of frozen phonon configurations
        "fp_sigmas":       {"Ba": 0.0757,   # Frozen phonon sigma values per atom type (A)
                            "Ti": 0.0893,   # sigma == sqrt(U_iso) (confirmed with abTEM author)
                            "Zr": 0.1050,   # Data from https://materials.springer.com/isp/crystallographic/docs/sd_1410590
                            "O":  0.0810},  # and from https://pubs.acs.org/doi/10.1021/acs.chemmater.9b04437 (for Zr)
        # PROBE SETTINGS
        "beam_energy":     200E3,           # Energy of the electron beam (eV)
        "convergence":     17.9,            # Probe semiangle of convergence (mrad)
        "tilt_mag":        5,               # Small-angle mistilt magnitude (mrad)
        "tilt_dir":        radians(45),     # Small-angle mistilt direction (deg from +x)
        # DETECTOR SETTINGS
        "max_batch":       200,             # Number of probe positions to propogate at once
        "max_angle":       40}              # Maximum detector angle (mrad)

# %% HARDWARE SETUP
if prms["device"] == "gpu":
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    dev = cupy.cuda.Device(prms["gpu_num"])
elif prms["device"] == "cpu":
    dev = cupy.cuda.Device(None)
else:
    raise TypeError('prms["device"] must be set to one of "gpu" or "cpu"')

# %% RUN
struct = aio.read(os.path.join(prms["path"], prms["filename"]))
with dev:
    detectors = [PixelatedDetector(max_angle=prms["max_angle"], resample="uniform")]
    for za_idx in prms["zas"]:
        start_time = timer()
        print("\nSetting up for " + str(za_idx) + " zone axis...", end=" ")

        # Calculate parameters for the projected cell
        _tmp_atoms = abuild.surface(struct, indices=za_idx, layers=1, periodic=True)
        a, b, c = _tmp_atoms.cell
        a = a[0]
        b = b[1]
        c = c[2]
        del _tmp_atoms  # Don't accidentally use the temporary surface

        thickness_multiplier = int((prms["thickness"] // c)) + 1
        atoms = abuild.surface(struct, indices=za_idx, layers=thickness_multiplier, periodic=True)
        atoms = orthogonalize_cell(atoms)
        atoms *= (prms["tiling"], prms["tiling"], 1)

        # Initial atom potential for grid matching; won't be directly used in sims
        potential = Potential(atoms,
                              sampling=prms["sampling"],
                              device=prms["device"],
                              projection="infinite",
                              parametrization="kirkland",
                              slice_thickness=prms["slice_thickness"])

        tilt = (prms["tilt_mag"]*sin(prms["tilt_dir"]),
                prms["tilt_mag"]*cos(prms["tilt_dir"]))
        probe = Probe(energy=prms["beam_energy"],
                      semiangle_cutoff=prms["convergence"],
                      device=prms["device"],
                      tilt=tilt)
        probe.grid.match(potential)

        if(any(angle < prms["max_angle"] for angle in probe.cutoff_scattering_angles)):
            warnings.warn("Scattering angle cutoffs < 200mrad; increase potential sampling")

        grid = GridScan(start=[0, 0],
                        end=[a, b],
                        sampling=probe.ctf.nyquist_sampling)

        save_delta = round(prms["thickness_step"] / prms["slice_thickness"])
        save_chunks = [(i, i + save_delta) for i in range(0, len(potential), save_delta)]
        measurements = [probe.validate_scan_measurements(detectors, grid) for chunk in save_chunks]

        fp = FrozenPhonons(atoms,
                           sigmas=prms["fp_sigmas"],
                           num_configs=prms["fp_configs"],
                           seed=prms["seed"])

        end_time = timer()
        elapsed = "{:0.2f}".format(end_time - start_time) + "s"
        print("Done!  Elapsed time was " + elapsed, end="\n\n")

        print("Beginning simulation...", end="\n\n")
        start_time = timer()
        num_fp_cfgs = len(fp)
        cfg_num = 0
        for atom_cfg in fp:
            cfg_num += 1
            print("Frozen phonon configuration " + str(cfg_num) + "/" + str(num_fp_cfgs) + ":")
            # Must rebuild the potential for each frozon phonon configuration
            potential = Potential(atom_cfg,
                                  sampling=prms["sampling"],
                                  device=prms["device"],
                                  storage="cpu",  # Store in RAM, otherwise can overload GPU memory
                                  projection="infinite",
                                  parametrization="kirkland",
                                  slice_thickness=prms["slice_thickness"])

            for indices, positions in grid.generate_positions(max_batch=prms["max_batch"]):
                waves = probe.build(positions)
                for chunk_idx, (slice_start, slice_end) in tqdm(enumerate(save_chunks),
                                                                total=len(save_chunks),
                                                                desc="Propogating",
                                                                unit="chunk"):
                    potential_slices = potential[slice_start:slice_end]
                    waves = waves.multislice(potential_slices, pbar=False)
                    for detector in detectors:  # Only one detector, but this supports multiple
                        new_measurements = detector.detect(waves)
                        grid.insert_new_measurement(measurements[chunk_idx][detector],
                                                    indices, new_measurements)
        end_time = timer()
        elapsed = "{:0.2f}".format(end_time - start_time) + "s"
        print("Finished simulating zone axis " + str(za_idx), ", elapsed time was " + elapsed)

        # %% EXPORT
        print("Exporting...", end=" ")
        export_path = os.path.join(prms["path"], "PACBED")
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        stack = []
        for i in range(len(measurements)):
            export_name = (f"{os.path.splitext(prms['filename'])[0]}_PACBED_" +
                           f"tilt{prms['tilt_mag']}mrad@{degrees(prms['tilt_dir'])}deg_" +
                           f"{str(za_idx)}_{str(int(prms['thickness']))}A_with_stepsize_" +
                           f"{str(int(prms['thickness_step']))}A.tif")
            xy = measurements[i][detectors[0]].sum((0, 1)).array
            stack.append(xy)
        tifffile.imwrite(os.path.join(export_path, export_name),
                         stack, photometric='minisblack')
        print("Done!")

        if prms["device"] == "gpu":  # Free the GPU memory for others
            # If this isn't done, GPU memory is only freed when the Python kernel is closed/reset
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

    print("\nSimulation complete!")
