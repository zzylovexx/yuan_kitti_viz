import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import yaml
import pandas as pd
from kitti_util import *
from matplotlib.lines import Line2D
import cv2
from PIL import Image
def affine_transform(projected_pts2d, trans_matrix): #(8,2) ,(2,3) 
    homo_projected_pts2d =np.concatenate((projected_pts2d, np.ones(projected_pts2d.shape[0])[:,None]),axis = 1)
    transformed_points = np.dot(homo_projected_pts2d, trans_matrix.T)
    return transformed_points[:, :2]
       
def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2

def draw_2d_box(image, left, top ,right, buttom):
     left=int(left)
     top = int (top)
     right = int (right)
     buttom = int(buttom)
     cv2.rectangle(image, (left,top),(right,buttom),(0,255,0),thickness=1)
     return image
def read_detection(path):
    if (os.path.getsize(path)==0):
        return 0
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score','trans_1',
                'trans_2','trans_3','trans_4','trans_5','trans_6','inv_tran_1','inv_tran_2','inv_tran_3','inv_tran_4','inv_tran_5','inv_tran_6']
#     df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
#     df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    df = df[df['type']=='Car']
    df.reset_index(drop=True, inplace=True)
    return df
# GUPNet/code/outputs_dir/origin_projection_affine_flip_camera_val_affine
detect_dir= '../GUPNet/code/outputs_dir/origin_1class_affine_val/data/'
save_dir = 'viz_outputs/origin_1class_affine_val/'
os.makedirs(save_dir, exist_ok=True)
for img_name in sorted(os.listdir(detect_dir)):
    img_id =int(img_name.replace('.txt',''))
    # img_id= int(img_id)
    print(img_id)

    calib = Calibration('../GUPNet/code/Dataset/KITTI/training/calib/%06d.txt'%img_id)
    # GUPNet/code/Dataset/KITTI/training/calib
    path_img = '../GUPNet/code/Dataset/KITTI/training/image_2/%06d.png'%img_id

    df = read_detection(detect_dir+'%06d.txt'%img_id)
    
    # GUPNet/code/outputs_dir/origin/data
    if (type(df)==int):
            continue

    # print(df.loc[0,['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
    image = cv2.imread(path_img)
    trans_matrix = np.array([df['trans_1'][0], df['trans_2'][0], df['trans_3'][0], 
                                 df['trans_4'][0], df['trans_5'][0], df['trans_6'][0]]).reshape((2,3))
    trans_inv = np.array([df['inv_tran_1'][0], df['inv_tran_2'][0], df['inv_tran_3'][0],
                                     df['inv_tran_4'][0],df['inv_tran_5'][0],df['inv_tran_6'][0]]).reshape((2,3))
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.transform((1280,384),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
    image = np.array(image)
    df.head()

    # print(len(df))


    ##############plot 3D box#####################
    for o in range(len(df)):
        
        corners_3d_cam2 = compute_3d_box_cam2(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
        affine_pts_2d  = affine_transform(pts_2d, trans_matrix)
        image = draw_projected_box3d(image, affine_pts_2d, color=(255,0,255), thickness=1)
        image = draw_2d_box(image, *df.loc[o, ['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
    cv2.imwrite(save_dir+str(img_name).replace('.txt','.png') , image)


    