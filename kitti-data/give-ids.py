"""
Feature extraction using deep models over the KITTI dataset.
Author: Benedikt Schröter & Tim Rosenkranz
Goethe University, Frankfurt am Main, Germany
"""

import pandas as pd
import numpy as np

infile = "annotations.csv"
outfile = "annotations_inc_id.csv"

column_names = ['filename', 'class', 'truncated', 'occluded', 'observation angle', \
                'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'length', \
                'xloc', 'yloc', 'zloc', 'rot_y']

df = pd.read_csv(infile, usecols=column_names)

df['id'] = np.NaN

for ii, row in df.iterrows():
    df.at[ii, 'id'] = ii

df.to_csv(outfile, index=False)
