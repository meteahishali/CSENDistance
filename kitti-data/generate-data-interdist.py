import pandas as pd
import numpy as np

def euclidean_distance_3D(x1, x2, x3, y1, y2, y3):
    return np.sqrt((x1 - y1)**2 + (x2 - y2)**2 + (x3 - y3)**2)

def euclidean_distance_2D(x1, x2, y1, y2):
    return np.sqrt((x1 - y1)**2 + (x2 - y2)**2)

infile = "annotations_inc_id.csv"
outfile = "annotations_interdistance.csv"

column_names = ['filename', 'class', 'truncated', 'occluded', 'observation angle', \
                'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'length', \
                'xloc', 'yloc', 'zloc', 'rot_y', 'id']

df = pd.read_csv(infile, usecols=column_names)

# New shape:
new_shape = ['filename', 'object_ids', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'obj distance 2D', 'obj distance 3D']
df2 = []

go_to = 10

for ii, row in df.iterrows():
    file = row['filename']
    id = ii
    xloc = row['xloc']
    yloc = row['yloc']
    zloc = row['zloc']

    df_buff = df[df['filename'] == file]
    df_buff = df_buff[df_buff['id'] > id]
    for jj, row2 in df_buff.iterrows():
        id2 = jj
        xloc2 = row2['xloc']
        yloc2 = row2['yloc']
        zloc2 = row2['zloc']

        inter_object_distance3D = euclidean_distance_3D(xloc, yloc, zloc, xloc2, yloc2, zloc2)
        inter_object_distance2D = euclidean_distance_2D(xloc, zloc, xloc2, zloc2)

        df2.append([file, (id, id2), xloc, yloc, zloc, xloc2, yloc2, zloc2, inter_object_distance2D, inter_object_distance3D])

    if ii%400 == 0:
        print(ii/407.50, '%')

df_out = pd.DataFrame(df2, columns=new_shape)
df_out.to_csv(outfile, index=False)
