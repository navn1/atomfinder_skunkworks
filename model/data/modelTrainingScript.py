'''
We are going to use a UNet-like neural network for semantic segmentation. 
In the semantic segmentation tasks we aim at categorizing every pixel in the image into background or atoms.
'''
import atomai as aoi
import numpy as np
import os, sys
from datetime import datetime

def makeModel(learningRate, X_train, y_train, X_test, y_test):
  #curDT = datetime.now()

 
  # Use nb_classes=1 for our training data. 
  # SegResNet is designed for such segmentation task. 
  # You can also try other availble architecture like 'Unet' or custom ones.
  model_semantic = aoi.models.Segmentor(model = 'SegResNet', nb_classes=1)

  print('Start training.')
  # You can use the much smaller training_cycles for testing, adjust it according to training convergence
  # Test changing loss function to mse instead of default ce
  model_semantic.fit(X_train, y_train, X_test, y_test, lr_scheduler=[learningRate], training_cycles=500, 
  plot_training_history = True, compute_accuracy = True, swa=True, filename = './data/model_'+ sys.argv[1][5:-4] + 'lr'+learningRate)
  
if __name__ == "__main__":
  data = np.load(os.getcwd() + '/' + sys.argv[1], allow_pickle=True)
  images_all = data['X_train']
  labels_all = data['y_train']
  images_test = data['X_test']
  labels_test = data['y_test']
  print('Data loaded.')
  for lr in [10^-6,(10^-6+10^-5)/2,10^-5,(10^-5+10^-4)/2,1*10^-4,(10^-4+10^-3)/2,1*10^-3]:
    makeModel(lr, images_all, labels_all, images_test, labels_test)
