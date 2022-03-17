# atomfinder_skunkworks
Atomfinder project for Informatics Skunkworks

**Purpose:**
We aim to create a model that accurately classifies atom positions within a given Scanning Electron Transmission Microscope (STEM) image of a crystal, that performs better than existing models in edge case scenarios such as when the STEM image consists of a crystal boundary or interface. 

**Folder details:**
The model folder contains the script to train the model, and a Dockerfile to specify how a docker image is to be built with the necessary dependencies (namely atomai). 
The simulation folder contains the script to train the model, and a Dockerfile to specify how a docker image is to be built with the necessary dependencies (namely abtem). 

**Workflows:**
There are workflows in the workflows folder configured such that when a change is committed and pushed to either the models and/or simulation folders, a docker image will be built and pushed to a private docker repository. 

Specifically, when there is a change pushed to the models folder, a workflow will be triggered to build a docker image that contains the script and its necessary dependencies, and then push that image to the docker repository under the "models" tag. 

Likewise, when there is a change pushed to the simulations folder, a workflow will be triggered to build a docker image that contains the script and its necessary dependencies, and then push that image to the docker repository under the "simulation" tag
