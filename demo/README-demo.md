# Demo
This demo has two different scripts. Both versions require a trained network (here CycleGAN) on data used for the synthesis on camera. A model state is then loaded into the networks to run a testing phase. This demo is primarily used for the Horse/Zebra datasets, because toy horses and zebras are easy to catch on camera. The following decription is therefore targeted to horse<->zebra synthesis.

## synth_on_cam
The synth_on_cam method is used to either create new datasets with the connected camera (if do_synth is set to False), or to also save the corresponding synthetic counterparts (if do_synth is set to True). While running, a window with the camera image is beeing shown. When hitting the keys 'SPACE', 'h' or 'p' one saves the image as a horse with or without synthetic zebra. Hitting 'ENTER' or 'z' saves zebra images and synthetic horses, respectively.

All images are beeing saved in the given 'synth_on_cam_dir' directory. Every new run of the script creates a new subdirectory 'try<Nr.>' to save all created real/fake horses/zebras.

The following parameters can or must be adopted:
  - do_synth: set to True or False to create and save synthetic images
  - synth_on_cam_dir: set the directory so save the cam_results
  - model_dir: set the directory of your model results, where your runs are beeing saved
  - checkpoint_name: set the specific run of your model
  - state: set the specific model states (epoch) you want to load

## synth_on_cam_live
This is the live version of synth_on_cam. This is not used for creating datasets or to save synthetic images, but rather to visualize the synthesis live. Starting the script creates a GUI with two shown images and three options as well as a Stop-button. The images are showing the video catched by the connected camera and the GUI is preset to 'no synth'. The other options are 'horse to zebra' and 'zebra to horse'. Pressing one of these other modes result in the right image showing the synthetic correspoding zebra or synthetic horse, depending on the mode. While the mode is beeing active, you can freely position different objects in front of the camera or e. g. hold the items in you hand.

Before starting the script, you need to install one additional package:
    
    pip install opencv-python==4.5.5.62
    pip install PySimpleGUI

The following parameters can or must be adopted:
  - show_on_laptop: set to False, when using a secondary screen.
  - img_size: you may need to adapt the image size to the resolution of your screen
  - cam_number: set it to the number of your cam (laptop cam is #0, additional cam is #1)
  - model_dir: set the directory of your model results, where your runs are beeing saved
  - checkpoint_name: set the specific run of your model
  - state: set the specific model states (epoch) you want to load

This script uses multi-threading with the ThreadPoolExecutor of concurrent.features to increase the fps.

### Troubleshooting
When using multiple monitors or trying to run the demo on other monitors with different resolutions, the position of the window and the sizes of the displayed images can be tricky. If possible, set the monitor's resolution to 1920x1080 and use it as your primary screen. You might have to resize the image (set img_size) additionally so that the GUI can be displayed completely. The location of the window may also need to be adapted.

Crashing the script may result in a hidden python script that blocks the camera. You may need to force quit this script. Restarting the kernel (the console) may also work.
