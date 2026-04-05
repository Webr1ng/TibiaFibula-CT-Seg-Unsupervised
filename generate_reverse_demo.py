import sys
sys.path.insert(0, '/data2/lyb/ankleSeg/TibiaFibula-CT-Seg-Unsupervised')

import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import pydicom

from src.segmentation.core import (
    preprocess_image, get_seed_points, region_growth,
    keep_largest_connected_components, morphology_optimization,
    separate_tibia_fibula
)

def segment(img):
    _, seeds = get_seed_points(img)
    grown = region_growth(img, seeds, threshold=20)
    kept = keep_largest_connected_components(grown, top_k=2)
    refined = morphology_optimization(kept)
    tibia, fibula = separate_tibia_fibula(refined, strategy='enhanced')
    return tibia, fibula

# Load a random test DICOM
dcm_files = glob.glob('/data2/lyb/ankleSeg/TibiaFibula-CT-Seg-Unsupervised/Data_new/testset/**/*.DCM', recursive=True)
random.seed(42)
ds = pydicom.dcmread(random.choice(dcm_files))

# Forward
img = preprocess_image(ds, window_center=300, window_width=800)
tibia_fwd, fibula_fwd = segment(img)

# Flipped
img_flip = np.fliplr(img)
tibia_rev, fibula_rev = segment(img_flip)

def overlay(base, tibia, fibula):
    rgb = np.stack([base, base, base], axis=-1)
    rgb[tibia > 0] = [255, 0, 0]    # red = tibia
    rgb[fibula > 0] = [0, 255, 0]   # green = fibula
    return rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(overlay(img, tibia_fwd, fibula_fwd))
axes[0].set_title('Forward Scan\n(Red=Tibia, Green=Fibula)', fontsize=12)
axes[0].axis('off')

axes[1].imshow(overlay(img_flip, tibia_rev, fibula_rev))
axes[1].set_title('Reversed Scan (Auto-corrected)\n(Red=Tibia, Green=Fibula)', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
out = '/data2/lyb/ankleSeg/TibiaFibula-CT-Seg-Unsupervised/Figure4-2_ReverseScan.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
