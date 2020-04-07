import os, sys

print("Starting...")

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import json
import random
import time
import pprint

import matplotlib.pyplot as plt

import run_nerf

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

#------------------------------------------------------------
basedir = './logs'
expname = 'fern_example'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf.config_parser()

args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, 'model_200000.npy')))
print('loaded args')

images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, 
                                                          recenter=True, bd_factor=.75, 
                                                          spherify=args.spherify)
H, W, focal = poses[0,:3,-1].astype(np.float32)

H = int(H)
W = int(W)
hwf = [H, W, focal]

images = images.astype(np.float32)
poses = poses.astype(np.float32)

if args.no_ndc:
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
else:
    near = 0.
    far = 1.
    

#------------------------------------------------------------
# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

bds_dict = {
    'near' : tf.cast(near, tf.float32),
    'far' : tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)


down = 4    # trade off resolution+aliasing for render speed
render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
render_kwargs_fast['N_importance'] = 0

c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix

#Start computing picture
#test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)
#img = np.clip(test[0],0,1)
#print("Close picture to start render videofile")
#plt.imshow(img)
#plt.show()

#------------------------------------------------------------
print("Rendering video")
down = 8 # trade off resolution+aliasing for render speed to make this video faster


#get just several first frames
max_frames = 10
skip_frames = 3
limit_frames = max_frames*skip_frames

frames = []

for i, c2w in enumerate(render_poses):
    if i >= limit_frames:
        break    # break here
    if i % skip_frames != 0:
        continue
    print("frame ", i+1,"/",render_poses.size, " limit: ", limit_frames+1)
    test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w[:3,:4], **render_kwargs_fast)
    frames.append((255*np.clip(test[0],0,1)).astype(np.uint8))
    
f = '_video.mp4'
print('done, saving ', f)
#imageio.mimwrite(f, frames, fps=30, quality=8)
imageio.mimwrite(f, frames, fps=30 / skip_frames, quality=9)
