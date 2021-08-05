# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:57:59 2021

@author: admin
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    # To do
    filter_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    filter_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    im_pad = np.zeros((im.shape[0]+2, im.shape[1]+2))
    x_offset = int(filter.shape[0]/2)
    y_offset = int(filter.shape[1]/2)
    im_pad[x_offset:im.shape[0]+x_offset,y_offset:im.shape[1]+y_offset] = im
    im_filtered = np.zeros(im.shape)
    b = filter
    for i in range(x_offset,im_pad.shape[0]-x_offset):
        for j in range(y_offset,im_pad.shape[1]-y_offset):
            a = im_pad[i-x_offset:i+x_offset+1, j-y_offset:j+y_offset+1]
            c = np.multiply(a,b)
            value = sum(sum(c))
            im_filtered[i-1][j-1] = value
    
    # im_filtered = im_filtered.astype(np.uint8)
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    grad_mag = np.zeros(im_dx.shape)
    grad_angle = np.zeros(im_dy.shape)
    for i in range(im_dx.shape[0]):
        for j in range(im_dx.shape[1]):
            value = (im_dx[i][j]**2 + im_dy[i][j]**2)**0.5
            grad_mag[i][j] = value
            grad_angle[i][j] = (np.arctan((im_dy[i][j])/(im_dx[i][j]+10**(-100))))
            if grad_angle[i][j] <0:
                grad_angle[i][j] = grad_angle[i][j]+np.pi
    # grad_mag = grad_mag.astype(np.uint8)
    # print(grad_mag)
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    tensor_size_x = int(grad_mag.shape[0]/cell_size)
    tensor_size_y = int(grad_mag.shape[1]/cell_size)
    theta = np.array([[165,180],[0,15],[15,45],[45,75],[75,105],[105,135],[135,165]])
    ori_histo = np.zeros((tensor_size_x, tensor_size_y, len(theta)-1))
    grad_angle = grad_angle*180/np.pi

    for i in range(tensor_size_x):
        for j in range(tensor_size_y):
            for p in range((i)*cell_size,(i+1)*cell_size):
                for q in range((j)*cell_size, (j+1)*cell_size):
                    if theta[0][0] <= grad_angle[p][q] < theta[0][1] or theta[1][0] <= grad_angle[p][q] < theta[1][1]:
                        ori_histo[i][j][0] = grad_mag[p][q] + ori_histo[i][j][0]
                    elif theta[2][0] <= grad_angle[p][q] < theta[2][1]:
                        ori_histo[i][j][1] = grad_mag[p][q] + ori_histo[i][j][1]
                    elif theta[3][0] <= grad_angle[p][q] < theta[3][1]:
                        ori_histo[i][j][2] = grad_mag[p][q] + ori_histo[i][j][2]
                    elif theta[4][0] <= grad_angle[p][q] < theta[4][1]:
                        ori_histo[i][j][3] = grad_mag[p][q] + ori_histo[i][j][3]
                    elif theta[5][0] <= grad_angle[p][q] < theta[5][1]:
                        ori_histo[i][j][4] = grad_mag[p][q] + ori_histo[i][j][4]
                    elif theta[6][0] <= grad_angle[p][q] < theta[6][1]:
                        ori_histo[i][j][5] = grad_mag[p][q] + ori_histo[i][j][5]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    ori_histo_normalized_size_x = ori_histo.shape[0] - (block_size - 1)
    ori_histo_normalized_size_y = ori_histo.shape[1] - (block_size - 1)
    ori_histo_normalized_size_z = 6 * (block_size**2)
    # Building a descriptor
    ori_histo_normal_descrp = np.zeros((ori_histo_normalized_size_x, ori_histo_normalized_size_y, ori_histo_normalized_size_z))
    ori_histo_normalized = np.zeros((ori_histo_normalized_size_x, ori_histo_normalized_size_y, ori_histo_normalized_size_z))
    blocks = 0
    for i in range(ori_histo_normalized_size_x):
        for j in range(ori_histo_normalized_size_y):
            for p in range(block_size):
                for q in range(block_size):
                    ori_histo_normal_descrp[i,j,blocks*6:(blocks+1)*6] = ori_histo[i+p,j+q,:]
                    blocks = blocks + 1
            blocks = 0
    # Normalizing of descriptor
    e = 0.001
    for i in range(ori_histo_normalized_size_x):
        for j in range(ori_histo_normalized_size_y):
            for k in range(ori_histo_normalized_size_z):
                ori_histo_normalized[i,j,k] = ori_histo_normal_descrp[i,j,k]/(np.sqrt(np.sum(ori_histo_normal_descrp[i,j]*ori_histo_normal_descrp[i,j]) + e**2))
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    filter_x, filter_y = get_differential_filter()
    im_filter_x = filter_image(im, filter_x)
    # im_filter_x_colr = cv2.applyColorMap(im_filter_x, cv2.COLORMAP_JET)
    
    # Uncomment this
    # cv2.imwrite('im_filter_x_grey.png', im_filter_x)
    # plt.axis('off')
    # plt.imshow(im_filter_x, cmap='jet', interpolation='nearest')
    # plt.savefig('im_filter_x.png', bbox_inches='tight', pad_inches=0)
    # Till here
    
    # cv2.imwrite('im_filter_x.png', im_filter_x_colr)
    im_filter_y = filter_image(im, filter_y)
    # im_filter_y_colr = cv2.applyColorMap(im_filter_y, cv2.COLORMAP_JET)
    
    # Uncomment this
    # cv2.imwrite('im_filter_y_grey.png', im_filter_y)
    # plt.imshow(im_filter_y, cmap='jet', interpolation='nearest')
    # plt.savefig('im_filter_y.png', bbox_inches='tight', pad_inches=0)
    # # cv2.imwrite('im_filter_y.png', im_filter_y_colr)
    # Till here

    grad_mag, grad_angle = get_gradient(im_filter_x, im_filter_y)
     # grad_mag_colr = cv2.applyColorMap(grad_mag, cv2.COLORMAP_JET)
    
    # Uncomment this
    # cv2.imwrite('im_gradient_mag_grey.png', grad_mag)
    # plt.imshow(grad_mag, cmap='jet', interpolation='nearest')
    # plt.savefig('im_gradient_mag.png', bbox_inches='tight', pad_inches=0)
    
    # # cv2.imwrite('im_gradient_mag.png', grad_mag_colr)
    # # plt.imshow(grad_angle, cmap='jet', interpolation='nearest')
    # # grad_angle_colr = cv2.applyColorMap(grad_angle, cv2.COLORMAP_JET)
    # cv2.imwrite('im_gradient_angle_grey.png', grad_angle)
    # plt.imshow(grad_angle, cmap='jet', interpolation='nearest')
    # plt.savefig('im_gradient_angle.png', bbox_inches='tight', pad_inches=0)
    # # cv2.imwrite('im_gradient_angle.png', grad_angle_colr)
    # Till here
    
    ori_histo = build_histogram(grad_mag, grad_angle, 8)
    # print(ori_histo.shape)
    # max_value = np.amax(ori_histo)
    # print(max_value)
    # ori_im = cv2.imread('cameraman.tif', 0)
    # for i in range(32):
    #     for j in range(32):
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 + int(4*ori_histo[i][j][0]/max_value), (j*8)+4), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 - int(4*ori_histo[i][j][0]/max_value), (j*8)+4), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 + int(0.86*4*ori_histo[i][j][0]/max_value), (j*8) + 4 + int(0.5*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 - int(0.86*4*ori_histo[i][j][0]/max_value), (j*8) + 4 - int(0.5*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 + int(0.5*4*ori_histo[i][j][0]/max_value), (j*8) + 4 + int(0.86*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 - int(0.5*4*ori_histo[i][j][0]/max_value), (j*8) + 4 - int(0.86*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 + int(0*4*ori_histo[i][j][0]/max_value), (j*8) + 4 + int(1*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 - int(0*4*ori_histo[i][j][0]/max_value), (j*8) + 4 - int(1*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 - int(0.5*4*ori_histo[i][j][0]/max_value), (j*8) + 4 + int(0.86*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 + int(0.5*4*ori_histo[i][j][0]/max_value), (j*8) + 4 - int(0.86*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 - int(0.86*4*ori_histo[i][j][0]/max_value), (j*8) + 4 + int(0.5*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    #         ori_im =  cv2.line(ori_im, ((i*8)+4, (j*8)+4), ((i*8)+ 4 + int(0.86*4*ori_histo[i][j][0]/max_value), (j*8) + 4 - int(0.5*4*ori_histo[i][j][0]/max_value)), (255,0,0),1)
    # cv2.imwrite('im_withHOG.png', ori_im)
    
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)
    # print(ori_histo_normalized.shape)
    
    # visualize to verify
    # visualize_hog(im, ori_histo_normalized, 8, 2)

    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.savefig('im_withNormHOG.png', bbox_inches='tight', pad_inches=0, dpi=255)
    plt.show()





def face_recognition(I_target, I_template):
    target_x = I_target.shape[0]
    target_y = I_target.shape[1]
    template_x = I_template.shape[0]
    template_y = I_template.shape[1]
    template_HOG = extract_hog(I_template)
    mod_value = sum(sum(sum(template_HOG)))
    # target_HOG = extract_hog(I_target)
    # print(target_HOG.shape)
    bounding_boxes = np.empty((0,3), int)
    heat_map = np.zeros((target_x-template_x, target_y-template_y))
    attempt = 0
    for i in range(target_x-template_x):
        for j in range(target_y-template_y):
            I_target_copy = I_target[i:i+template_x, j:j+template_y]
            target_HOG = extract_hog(I_target_copy)
            heat_map_value = sum(sum(sum(target_HOG*template_HOG)))/(sum(sum(sum(target_HOG*target_HOG))) * mod_value)
            heat_map[i, j] = heat_map_value
            if heat_map_value >= 0.001 and i < 40:
                print('Bounding box detected')
                bounding_boxes_row = np.array([j, i, heat_map_value])
                bounding_boxes = np.vstack((bounding_boxes, bounding_boxes_row))
        print(i)
    plt.imshow(heat_map, cmap='hot', interpolation='nearest')
    plt.savefig('heat_map_detection.png', bbox_inches='tight', pad_inches=0, dpi=200)
    bounding_boxes_all = bounding_boxes.copy()
    bounding_box_supre = np.empty((0,3), int)
    while bounding_boxes.size != 0:
        current_max = bounding_boxes[:,2].max()
        max_index = np.where(bounding_boxes == current_max)[0][0]
        max_x = bounding_boxes[max_index][0]
        max_y = bounding_boxes[max_index][1]
        bounding_box_supre = np.vstack((bounding_box_supre, bounding_boxes[max_index]))
        bounding_boxes = np.delete(bounding_boxes, (max_index), axis=0)
        copy_bounding_boxes = bounding_boxes.copy()
        for i in range(len(bounding_boxes)):
            x_dist = min((max_x+50), (bounding_boxes[i][0]+50)) - max(max_x, bounding_boxes[i][0])
            y_dist = min((max_y+50), (bounding_boxes[i][1]+50)) - max(max_y, bounding_boxes[i][1])
            if x_dist > 0 and y_dist > 0:
                area = x_dist * y_dist
                if area > 1000:
                    area_index = np.where(copy_bounding_boxes == bounding_boxes[i])[0][0]
                    copy_bounding_boxes = np.delete(copy_bounding_boxes, (area_index), axis=0)
                    # print('Deleted row')
        bounding_boxes = copy_bounding_boxes.copy()
    return  bounding_box_supre, bounding_boxes_all


def visualize_face_detection(I_target,bounding_boxes,box_size, num):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.savefig('face_detected_{}.png'.format(num), dpi=200)
    plt.show()




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    # mxn  face template

    bounding_boxes, bounding_boxes_all = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0], 1)
    
    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    # visualize_face_detection(I_target_c, bounding_boxes_all, I_template.shape[0], 2)
    #this is visualization code.
