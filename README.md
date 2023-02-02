# TODO
This repos is in contruction...

The base form is already finished, but this README, test.py and used_packages/ need to be worked on.
Also, some of the python functions are going to be relocated behind a file. So only train/test/registrate are the visible python files.

# I2I-Synthesis
I2I-Synthesis is a Image-to-Image Synthesis repository containing the CycleGAN framework. In addition to the training/testing of networks based on MR/CT/CBCT or natural (RGB) images, this repos also provides a live-on-cam synthesis script in [/demo](https://github.com/nilsFrohwitter/I2I-Synthesis/tree/main/demo).

All of this was created under the work at the [DFKI](https://www.dfki.de/en/web).

# Image Registration
For the evaluation of the influence of MR-to-CT synthesis in MR-to-CT image registration, the registrate.py script was created. This python script makes use of the [ANTsPy](https://antspy.readthedocs.io/en/latest/) registration framework and performs the registration with the methods 'Rigid', 'SyN', 'SyNRA' and 'ElasticSyN' between case0:MR-CT, case1:Synthetic_CT-CT and case2:MR-Synthetic_MR. 

To run the registration script, three parameters needs to be adopted:
    <pat>: This was the patient ID given as the suffix in the data directory. Either specify the data with this suffix or set it to None.
           This way all data in this directory is getting registrated. 
