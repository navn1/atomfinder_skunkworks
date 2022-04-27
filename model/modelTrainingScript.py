'''
We are going to use a UNet-like neural network for semantic segmentation. 
In the semantic segmentation tasks we aim at categorizing every pixel in the image into background or atoms.
'''
import atomai as aoi
import numpy as np
import os, sys
from datetime import datetime

def makeModel():
  curDT = datetime.now()
  data = np.load(os.getcwd() + '/' + sys.argv[1], allow_pickle=True)
  images_all = data['X_train']
  labels_all = data['y_train']
  images_test = data['X_test']
  labels_test = data['y_test']
  print('Data loaded.')
 
  # Use nb_classes=1 for our training data. 
  # SegResNet is designed for such segmentation task. 
  # You can also try other availble architecture like 'Unet' or custom ones.
  model_semantic = aoi.models.Segmentor(model = 'SegResNet', nb_classes=3)

  print('Start training.')
  # You can use the much smaller training_cycles for testing, adjust it according to training convergence
  model_semantic.fit(images_all, labels_all, images_test, labels_test, training_cycles=500, 
  plot_training_history = False, compute_accuracy = True, swa=True, filename = 'model_'+ curDT.strftime("%m%d_%H%M"))
  
if __name__ == "__main__":
    makeModel()