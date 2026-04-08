import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

flow_path = "/egr/research-sprintai/baliahsa/projects/SecMI-LDM/dataset/Datasets-Vision/Vehicle/optical_flow_224h/P716/R2+2b-vehicle_door_open_hazard+nback_task"

savefolder = "arrow_visualizations2"
os.makedirs(savefolder, exist_ok=True)

files = sorted([f for f in os.listdir(flow_path) if f.endswith(".npy")])

for f in files[113:116]:
    flow = np.load(os.path.join(flow_path, f)).astype(np.float32)  # shape HxWx2

    # Downsample for arrows (too many arrows are messy)
    step = 16
    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    
    fx = flow[y, x, 0]
    fy = flow[y, x, 1]

    plt.figure(figsize=(10,10))
    plt.imshow(np.zeros((h, w)), cmap='gray')  # blank background
    plt.quiver(x, y, fx, fy, color='red', angles='xy', scale_units='xy', scale=1, width=0.003)
    plt.axis('off')
    plt.title(f)
    plt.savefig(os.path.join(savefolder, f"{f[:-4]}_arrows.png"), bbox_inches='tight', pad_inches=0)
    plt.close()