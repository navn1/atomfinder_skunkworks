#Test python file - Replace this file with the real python simulation script!
import sys
import ase
import abtem
import json
from ase.visualize import view
from ase.io import read
from abtem.visualize import show_atoms
import matplotlib.pyplot as plt
from ase.build import surface
from abtem import *
def structure_prep():
  file = read(sys.argv[1])
  file.center()

  #Volume : 5,5,60 nm
  XY = 50/(file.get_cell_lengths_and_angles()[0])
  Z = 600/(file.get_cell_lengths_and_angles()[2])

  mindices = int(sys.argv[2])
  i1 = (int)(mindices%10)
  mindices/=10
  i2 = (int)(mindices%10)
  mindices/=10
  i3 = (int)(mindices%10)
  si_100 = surface(file, indices=(i1, i2, i3), layers=int(Z), periodic=True)
 
  si_100*=(int(XY),int(XY),1)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
  #show_atoms(si_100, plane='xz', ax = ax1)
  #show_atoms(si_100, plane='yz', ax = ax2)

  from abtem.structures import orthogonalize_cell
  si_o = orthogonalize_cell(file)

  oneSi100 = surface(file, indices=(1,0,0), layers = int(Z))
  si100_o = orthogonalize_cell(oneSi100)
  si100_o *=(int(XY),int(XY),1)
  print(si100_o.get_positions()) #gets atomic positions and coordinates of orthogonalized cell
 
  return si100_o


def simulate(struct_o): 
  from abtem.temperature import FrozenPhonons
  sigmadict = json.loads(sys.argv[4])
  fp = FrozenPhonons(struct_o, num_configs = int(sys.argv[3]), sigmas = sigmadict)


  probe = Probe(energy=200e3, semiangle_cutoff=24.5, defocus=0, device='gpu') # calculate the wave function of electron probe
  potential = Potential(fp, gpts=512, slice_thickness=2, projection='infinite', parametrization='kirkland', device ='gpu')

  from abtem.scan import GridScan
  pixelsize = sys.argv[5] # probe scan step size in angstrom

  slicenum = 10 # chose how many slices of potential to be integrated for the simulation. i.e. choose sample thickness = slicenum * slice thickness here
  potential_slice = potential[slicenum]
  detector = AnnularDetector(inner=100, outer=350) # define the detector that collects eletrons and generate the final STEM image. 
  gridscan = GridScan(start=[0, 0], end=[50, 50], sampling=(float)(pixelsize)) # define the scan grid
  measurement_files = probe.scan(gridscan, [detector], potential_slice, pbar=True) # run simulation

  from abtem.measure import Measurement

  measurement = Measurement.read(measurement_files)
  new_measurement = measurement.tile((5, 5))
  measurement.save_as_image(sys.argv[1][:sys.argv[1].index('.')] + '.tif') #saves simulated image in your directory


if __name__ == "__main__":
    simulate(structure_prep())
