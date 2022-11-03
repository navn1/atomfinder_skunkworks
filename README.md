# atomfinder_skunkworks
Atomfinder project for Informatics Skunkworks

**Purpose:**

We aim to create a model that accurately classifies atom positions within a given Scanning Electron Transmission Microscope (STEM) image of a crystal, that performs better than existing models in edge case scenarios such as when the STEM image consists of a crystal boundary or interface. 

**Folder details:**
- The simulation folder contains the script to simulate STEM images for training given crystal CIF files, and a Dockerfile to specify how a docker image is to be built with the necessary dependencies (namely abtem), and the simulation results. 
- The train folder contains the script to train the model, and a Dockerfile to specify how a docker image is to be built with the necessary dependencies (namely atomai). 
- The model folder contains necessary scripts to load the model structure and the best model weights. 
- The evaluation folder contains the example jupyter notebook used to evaluate models, and all the calculated evaluation metrics for this model. 

**Workflows:**

There are workflows in the workflows folder configured such that when a change is committed and pushed to either the models and/or simulation folders, a docker image will be built and pushed to a private docker repository. 

Specifically, when there is a change pushed to the train folder, a workflow will be triggered to build a docker image that contains the script and its necessary dependencies, and then push that image to the docker repository under the "train" tag. 

Likewise, when there is a change pushed to the simulations folder, a workflow will be triggered to build a docker image that contains the script and its necessary dependencies, and then push that image to the docker repository under the "simulation" tag

**How to Use the Model:**

Option1: Load the model by downloading this package.Then: 

` os.chdir('atomfinder_skunkworks/model')`<br />
`from app import *`<br />
`inputdict = {'image':array, 'change_size':1, 'glfilter_sigma':3}`<br />
`coordinates = run(inputdict)`

Option2: Load the model using **Foundry**. You can also load the training dataset from Foundry. <br />
See more info about Foundry on their [website](https://foundry-ml.org/). <br />
See example usage of Foundry in this notebook. <a href="https://colab.research.google.com/drive/1iL0G1FusBX4ToBw9Y1XX_H4fgNExp5yw?usp=sharing****">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
