'''
The input cif file and should be under the current working directory. 

python testing.py tiffile miller_indices pixelsize_angstrom
Example: python testing.py Silicon.cif 100 0.4

Created: Shreya Jagadeshwaran
Modified: Jingrui Wei
'''
import os
import sys
import ase
import abtem
#import json
import numpy as np
from ase.io import read
from ase.build import surface
from abtem import *
import matplotlib.pyplot as plt

def structure_prep():
  file = read(os.getcwd() + '/' + sys.argv[1])
  file.center()

  #Volume : 5,5,60 nm, use 5nm for test now
  XY = 10/(file.cell.cellpar()[0])
  Z = 12/(file.cell.cellpar()[2])

  mindices = int(sys.argv[2])
  i1 = (int)(mindices%10)
  mindices/=10
  i2 = (int)(mindices%10)
  mindices/=10
  i3 = (int)(mindices%10)
  stucture = surface(file, indices=(i1, i2, i3), layers=int(Z), periodic=True)
 
  stucture*=(int(XY),int(XY),1)

  from abtem.structures import orthogonalize_cell 
  stucture_o = orthogonalize_cell(stucture)
 
  return stucture_o


def simulate(struct_o): 
  from abtem.temperature import FrozenPhonons  
  from abtem.measure import Measurement

  # To keep things simple, we will ignore the thermal vibration here. 
  #fp = FrozenPhonons(struct_o, num_configs = int(sys.argv[3]), sigmas = sigmadict)

  probe = Probe(energy=200e3, semiangle_cutoff=24.5, defocus=0, device='gpu') # calculate the wave function of electron probe
  potential = Potential(struct_o, sampling = 0.01, slice_thickness=2, projection='infinite', parametrization='kirkland', device ='gpu')

  from abtem.scan import GridScan
  pixelsize =(float)(sys.argv[3]) # probe scan step size in angstrom, varies between 0.05, 0.15, 0.25, 0.35, 0.45 angstrom

  # integrate potential and generate image for thickness of 12,24,36,48,60 nm
  # Go thickness dowm to taking use of intermediate result in memory
  slicelist = [element * len(potential)//5 for element in [5,4,3,2,1]]
  images = []
  #print(len(potential), slicelist)
  for slicenum in slicelist:
    potential_slice = potential[:slicenum]
    print('Potential integrated for ', str(slicenum*0.4), ' nm') 
    detector = AnnularDetector(inner=100, outer=350) # define the detector that collects eletrons and generate the final STEM image. 
    # define the scan grid, use fixed grid size 100 for uniform image size    
    gridscan = GridScan(start = [0,0], end = [pixelsize*100,pixelsize*100], gpts=(100,100), sampling=pixelsize) # define the scan grid

    measurement_files = probe.scan(gridscan, [detector], potential_slice, pbar=True) # run simulation
    images.append(measurement_files.array)
  
  #gets atomic x,y coordinates of orthogonalized cell in angstrom
  posxy = np.unique(np.round(struct_o.get_positions()[:,:2],4), axis = 0)
  cropped = filter(lambda point: (point[0] < pixelsize*100) & (point[1] < pixelsize*100), posxy)
  cropped = np.array(list(cropped))
  # save the simulated images and coordinates in a single npz file
  np.savez(sys.argv[1][:-4] + sys.argv[2] + '_px'+ str(int(pixelsize*100)) + 'pm.npz', images = images, coordinates = cropped)
  print('All simulated results saved. ')

  # Make visulization figure to check the result
  fig = plt.figure(figsize = (10,10))
  plt.imshow(images[-1])
  plt.scatter(cropped[:,1]/pixelsize, cropped[:,0]/pixelsize ,c = 'r', s = 1)
  fig.savefig(sys.argv[1][:-4] + sys.argv[2] + '_px'+ str(int(pixelsize*100)) + 'pm.tif')


if __name__ == "__main__":
    simulate(structure_prep())
