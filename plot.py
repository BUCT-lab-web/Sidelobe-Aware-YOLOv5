import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

def plot_results(file='path/to/results.csv', dir=''):
    save_dir = Path(file).parent if file else Path(dir)
    # fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    # ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            # for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
            y = data.values[:, 1]
            # y[y == 0] = np.nan  # don't show zero values
            ax.plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
            ax.set_title('CIOU-'+s[1], fontsize=12)
            
            # fig.savefig(save_dir  /  s[j]  +'.png', dpi=200)
            # plt.close()
            # if j in [8, 9, 10]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    # ax[1].legend()
    fig.savefig(save_dir / 'train-box_loss.png', dpi=200)
    plt.close()

Path_csv = './runs/sarship/newdata/test1-3/retrain2/results.csv' 
plot_results(Path_csv)