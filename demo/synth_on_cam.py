import cv2
from options.test_options import TestOptions
from models.init import *
import torch
import os
from visualization import show_live
from savefile import save_image_live
import numpy as np

# dir where the cam frames and their synthetic images are saved 
synth_on_cam_dir = r"C:\Users\Nils\Documents\Synthesis CT MR\cam_results"
index_run = 0
while True:
    if os.path.exists(os.path.join(synth_on_cam_dir, 'try' + str(index_run))):
        index_run += 1
    else:
        # cam_frames_dir = os.path.join(synth_on_cam_dir, 'try' + str(index_run), 'cam_frames')
        frames_dir = os.path.join(synth_on_cam_dir, 'try' + str(index_run))
        for name in ['real_horse', 'real_zebra', 'fake_horse', 'fake_zebra']:
            # os.makedirs(os.path.join(cam_frames_dir, name))
            os.makedirs(os.path.join(frames_dir, name))
        break

# get testing options
opt = TestOptions().parse(do_print=False)  
opt.batch_size = 1
opt.input_nc = 3
opt.output_nc = 3

# set checkpoint of saved models
model_dir = r"C:\Users\Nils\Documents\Synthesis CT MR\results\CycleGAN Pferd_Zebra"
checkpoint_name = '22-03-13_22-45-49_f25fdcc4-a316-11ec-bec1-ac1f6bf5ab70'
state = r'checkpoints\epoch_100_03-14_07-21-25.pth'
opt.checkpoint_dir = os.path.join(opt.save_dir, checkpoint_name, state)
opt.name = checkpoint_name

# loading and preparing model
model = create_model(opt)
model.load_model()
model.eval()
print('model loaded')

# TODO: this should be in the cam part...
# test_slice = {'A': test_vol_data['A'][:, :, :, sl].unsqueeze(0),
#              'B': test_vol_data['B'][:, :, :, sl].unsqueeze(0),
#              'A_paths': test_vol_data['A_paths'], 'B_paths': test_vol_data['B_paths']}
# model.set_input(test_slice)
# with torch.no_grad():
#     model.forward()

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    if frame.shape[0] != frame.shape[1]:
        x = frame.shape[0]
        y = frame.shape[1]
        min_size = np.min((frame.shape[0], frame.shape[1]))
        # print('min is ' + str(min_size))
        frame = frame[int(np.floor((x-min_size)/2)):int(np.floor(min_size+(x-min_size)/2)), int(np.floor((y-min_size)/2)):int(np.floor(min_size+(y-min_size)/2)), :]
        
    
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32 or k%256 == 104 or k%256 == 112:
        # SPACE, 'h' or 'p' pressed for horse/Pferd
        img_name = "horse" + str(index_run) + "_frame_{}.png".format(img_counter)
        img_dir = os.path.join(frames_dir, 'real_horse', img_name)
        cv2.imwrite(img_dir, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        
        # now for the synth part:
        # image = cv2.imread(img_dir)
        # with torch.no_grad():
        #     model.set_input_live(image, 'horse')
        #     model.forward_live('horse')
        #     save_image_live(model.fake_b, frames_dir, img_name, 'horse')
        #     show_live(model.fake_b, 'fake_zebra')
        

    elif k%256 == 13 or k%256 == 122:
        # ENTER or 'z' pressed for zebra
        img_name = "zebra" + str(index_run) + "_frame_{}.png".format(img_counter)
        img_dir = os.path.join(frames_dir, 'real_zebra', img_name)
        cv2.imwrite(img_dir, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        
        # now for the synth part:
        # image = cv2.imread(img_dir)
        # with torch.no_grad():
        #     model.set_input_live(image, 'zebra')
        #     model.forward_live('zebra')
        #     save_image_live(model.fake_a, frames_dir, img_name, 'zebra')
        #     show_live(model.fake_a, 'fake_pferd')

    elif k==-1:
        continue

    else:
        print('You pressed %d (0x%x), LSB: %d (%s)' % (k, k, k % 256,
                                                       repr(chr(k%256)) if k%256 < 128 else '?'))
        print('This is not a valid key for horse (SPACE, h, p) or zebra (ENTER, z)!')

cam.release()

cv2.destroyAllWindows()
