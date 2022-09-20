# %%
import ase.io
import ase.visualize

# Loading a cif file with ASE is extremely straightforward, just call ase.io.read on the file path
cif_path = r"C:\Users\charles\Documents\GitHub\tem_simulation_examples\Silicon.cif"

cif_unit_cell = ase.io.read(cif_path)

ase.visualize.view(cif_unit_cell)

# %%
from ase import Atoms

# Alternatively, we can directly create a model using the ase.Atoms constructor
ase_unit_cell = Atoms("BaTiO3",                           # Unit cell chemistry
                      cell=[4.04, 4.04, 4.04],            # a, b, c lattice parameters (A)
                      pbc=True,                           # Periodic boundary conditions
                      scaled_positions=[(0, 0, 0),        # Atomic positions (fractional)
                                        (0.5, 0.5, 0.5),  # (same order as chemistry from 1st line)
                                        (0.5, 0.5, 0),
                                        (0.5, 0, 0.5),
                                        (0, 0.5, 0.5)])

ase.visualize.view(ase_unit_cell)

# %%
import ase.build

# Now we'll tell ASE to turn that unit cell into a 110 surface
atoms = ase.build.surface(ase_unit_cell, (1, 1, 0), layers=10, periodic=True)

# And finally we'll multiply out our surface to the dimensions we want
# We could multiply in Z here instead of doing layers=10 above, but it can cause bugs
model = atoms * (2, 2, 1)

ase.visualize.view(model)

# %%
from abtem.temperature import FrozenPhonons
from abtem import show_atoms

# Next up we'll generate some frozen phonon configurations for that model
fps = FrozenPhonons(model,                 # The model to generate phonon configus for
                    sigmas={"Ba": 0.0757,  # Debye-Waller params (sqrt(Uiso), A)
                            "Ti": 0.0893,
                            "O":  0.0810},
                    num_configs=30)        # Number of frozen phonon configs

atoms_conf = next(iter(fps))
show_atoms(atoms_conf)

# %% 
from abtem import Potential

# To run a simulation, we need to generate potential slices from our model
potential = Potential(fps,                         # The model to slice (Atoms object also works)
                      sampling=0.04,               # Sampling (1/A)
                      projection="infinite",       # Infinite is typical, finite is slower
                      parametrization="kirkland",  # Radial dependence of potential
                      slice_thickness=1,           # Slice thickness in A
                      device="gpu")

potential.project().show()

# %%
from abtem.waves import Probe

# Next let's define the probe
probe = Probe(energy=200E3,           # Energy (eV)
              semiangle_cutoff=17.9,  # Semiangle (mrad)
              device="gpu")

# We can set some aberations too, using the probe CTF
probe.ctf.set_parameters({"astigmatism": 7,
                          "astigmatism_angle": 155,
                          "coma": 300,
                          "coma_angle": 155,
                          "Cs": -5000})

# Ensure that the sampling of the probe matches that of the potential
probe.grid.match(potential)

probe.show()

# %%
from abtem.scan import GridScan

# A logical next step is to define our scan grid; I'll scan the whole projected cell
scan = GridScan(start=[0, 0],                             # Start coordinates, A
                end=potential.extent,                     # potential.extent = scan everything
                sampling=probe.ctf.nyquist_sampling*0.9)  # A reasonable sampling resolution

ax, im = potential.project().show()
scan.add_to_mpl_plot(ax)

# %%
from abtem.detect import FlexibleAnnularDetector
from abtem.detect import PixelatedDetector

# Finally, to get legible results we need some simulated detectors
# A FlexibleAnnularDetector lets us define the bounds of integration later, after the simulation
fad = FlexibleAnnularDetector()

# A pixelated detector can be used to simulat 4D STEM data, or CBED/PACBED images
pd = PixelatedDetector(max_angle=100,       # Detector angular limit (mrad)
                       resample="uniform")  # Resample to have square pixels

# We put the detectors in a list so that the simulation will detect on all of them
detectors = [fad, pd]

# %%
import cupy

# Now that everything is set up, we can run the multislice simulation and get a measurement
# We call the scan function of the probe we set up earlier to do this
with cupy.cuda.Device(0):
    measurement = probe.scan(scan,
                             detectors,
                             potential,
                             pbar=False)  # The progress bar doesn't work in the current version

# %%

# Now we can take a closer look at the measurements we took
fad_measurement = measurement[0]
pd_measurement = measurement[1]

# We can get, for example, BF and HAADF images from the same FAD
bf = fad_measurement.integrate(0, 20)
haadf = fad_measurement.integrate(70, 200)

# And it's a good idea to do a 4x interpolation and tiling to get a large, smooth-looking image
haadf = haadf.interpolate(tuple([x/4 for x in scan.sampling]), kind="fft").tile((3, 3))
bf = bf.interpolate(tuple([x/4 for x in scan.sampling]), kind="fft").tile((3, 3))

# Let's take a look at these images
haadf.show()
bf.show()

# %%

# We can also look at diffraction data from this scan, since we have a 4D dataset saved to the PD
# Let's take the average along the scan dimensions and look at the PACBED; it won't look very good
# since the simulation conditions weren't optimal for it

pd_measurement.sum((0, 1)).show(cmap='inferno', power=0.3)