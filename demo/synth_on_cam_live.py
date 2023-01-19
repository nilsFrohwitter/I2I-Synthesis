import PySimpleGUI as sg
import cv2
import numpy as np
from options.test_options import TestOptions
from models.init import create_model
import torch
import os
import concurrent.futures


"""
This is the live version of synth_on_cam.
Images on cam and synthetic images are not saved, but rather rapidly visualized.

This is only tested in Windows.
"""

print('Start of the live demo')
# %% Adjustable parameters for live-demo:
show_on_laptop = False      # True for Laptop and False for second screen: Be sure to connect the second screen to the left of your laptop!
if show_on_laptop:
    img_size = 680          # e. g. 680 for laptop
else:
    img_size = 920          # e. g. 920 for 1920x1080 screen

sg_theme = 'LightGreen'     # 'Material2' or 'LightGreen'
window_name = 'live demo for horse-to-zebra synthesis'
cam_number = 1  # specifies which cam cv2 uses (default is 0)

"""Use pretrained network for live-demo, dataset: pferd_zebra_one2one_new
Set the name of the run and the state directory to load the net."""
checkpoint_name = '22-03-17_14-51-55_681847be-a5f9-11ec-9be4-ac1f6b9e87fe'
state = r'checkpoints\epoch_200_03-17_20-46-49.pth'  # note the windows-directory

# %% the rest must not be changed
# get testing options
opt = TestOptions().parse(do_print=False)
opt.batch_size = 1
opt.input_nc = 3  # since we are using a standard cam, we have RGB images
opt.output_nc = 3

# specify where the checkpoints of the learned models are located
model_dir = r"C:\Users\Nils\OneDrive - dfki.de\Dokumente\Synthesis CT MR\results\CycleGAN Pferd_Zebra"

opt.checkpoint_dir = os.path.join(opt.save_dir, checkpoint_name, state)
opt.name = checkpoint_name

# loading and preparing model
model = create_model(opt)
model.load_model()
model.eval()
print('model loaded')

# %%
sg.theme(sg_theme)
# define the window layout
layout = [
        # [sg.Text("Synthesis Demo", size=(80, 1), justification='right')],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Radio("no synth", "Radio", True, size=(10, 1), key="-NOSYNTH-")],
        [sg.Radio("horse to zebra", "Radio", size=(12, 1), key="-H2Z-")],
        [sg.Radio("zebra to horse", "Radio", size=(12, 1), key="-Z2H-")],
        [sg.Button("Stop", size=(6, 1))],
]

# Create the window and show it without the plot
if show_on_laptop:
    window = sg.Window(window_name, layout, location=(0, 0), finalize=True)
else:
    window = sg.Window(window_name, layout, location=(-1500, 50), finalize=True)
window.Maximize()


# %%
def process_images(values, frame):
    frame = frame[:, 80:560, :]
    frame_show = cv2.resize(frame, [img_size, img_size], interpolation = cv2.INTER_LANCZOS4)   # INTER_CUBIC, INTER_AREA, INTER_LINEAR, INTER_NEAREST, INTER_LANCZOS4

    if values["-NOSYNTH-"]:
        # frame = np.hstack([frame_show, frame_show])
        frame = np.hstack([frame_show, frame_show])
    elif values["-H2Z-"]:
        with torch.no_grad():
            model.set_input_live(frame, 'horse')
            model.forward_live('horse')
            img = model.fake_b / 2 + 0.5
            img = cv2.resize(np.transpose(img.squeeze(0).cpu().numpy(), (1, 2, 0)), [img_size, img_size], interpolation = cv2.INTER_LANCZOS4)
            frame = np.hstack([frame_show, np.round(img * 255)])

    elif values["-Z2H-"]:
        with torch.no_grad():
            model.set_input_live(frame, 'zebra')
            model.forward_live('zebra')
            img = model.fake_a / 2 + 0.5
            img = cv2.resize(np.transpose(img.squeeze(0).cpu().numpy(), (1, 2, 0)), [img_size, img_size], interpolation = cv2.INTER_LANCZOS4)
            frame = np.hstack([frame_show, np.round(img * 255)])

    imgbytes = cv2.imencode(".png", frame)[1].tobytes()

    window["-IMAGE-"].update(data=imgbytes)


# %%
cam = cv2.VideoCapture(cam_number)

thread_counter = 0
max_treads = 8
first_round = True
future = [i for i in range(max_treads)]

with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
    while True:
        event, values = window.read(timeout=100)
        thread_counter += 1

        if event == "Stop" or event == sg.WIN_CLOSED:
            break

        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # first start of multiple threads
        if first_round:
            executor.submit(process_images, values, frame)
            if thread_counter == max_treads:
                first_round = False
                thread_counter = 0
            continue
        # now multiprocessing after "initialization"
        else:
            executor.submit(process_images, values, frame)
            if thread_counter == max_treads:
                thread_counter = 0

window.close()

cam.release()

cv2.destroyAllWindows()
