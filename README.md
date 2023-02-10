# TODO
This repos is in contruction...

The base form is already finished, but this README, test.py and used_packages/ need to be worked on.
Also, some of the python functions are going to be relocated behind a file. So only train/test/registrate are the visible python files.

# I2I-Synthesis
I2I-Synthesis is a Image-to-Image Synthesis repository containing the CycleGAN framework. In addition to the training/testing of networks based on MR/CT/CBCT or natural (RGB) images, this repos also provides a live-on-cam synthesis script in [demo](/demo) which is used to synthesize between horse and zebra toys, held in front of the camera.

All of this was created under the work at the [DFKI](https://www.dfki.de/en/web).

## Installation
The training was done on an cluster with Linux OS, Ubuntu 18.04.01, using NVIDIA enroot images.
For a detailed list of used packages, see [packages](used%20packages/packages.txt).

To run the script in a conda envirement (this requires anaconda), use following installation script:


## Training
To adapt the options of the model and of the training, see [base_options](options/base_options) and [train_options](options/train_options).
After adaption and installation of the requirements, run the training with
      
      python train.py

## Testing
To adapt the options of the testing, see [test_options](options/test_options).
After adaption and installation of the requirements, run the testing with
      
      python test.py

# Image Registration
For the evaluation of the influence of MR-to-CT synthesis in MR-to-CT image registration, the registrate.py script was created. This python script makes use of the [ANTsPy](https://antspy.readthedocs.io/en/latest/) registration framework and performs the registration with the methods:

1. 'Rigid'
2. 'SyN'
3. 'SyNRA'
4. 'ElasticSyN' 

for the registration tasks:
1. MR-CT (case0)
2. Synthetic_CT-CT (case1)
3. MR-Synthetic_MR (case2). 

Before running the registration, install the additional antspy package:

      pip install antspyx

To run the registration script, four parameters needs to be adopted:

   - 'pat': This was the patient ID given as the suffix in the data directory. Either specify the data with this suffix or set it to None.
      This way all data in this directory is getting registrated. 
   - 'run': Specify the name of the run from the model you want to use.
   - 'image_origin': Set it to \[val_images|test_images\]. val_images is recommended, because of some hard coded parts.
   - 'dir_base': Specify the root for runs, data and results

As a result, csv files with detailed registration results (slice-wise) and averaged results are saved. Furtheremore, plots with comparisons of registration performances are beeing created. 
