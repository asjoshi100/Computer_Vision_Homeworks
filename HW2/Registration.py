import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import pandas as pd
import time


def find_match(img1, img2):
    # To do
    sift_temp = cv2.xfeatures2d.SIFT_create()
    sift_targ = cv2.xfeatures2d.SIFT_create()
    kp_temp, des_temp = sift_temp.detectAndCompute(img1,None)
    kp_targ, des_targ = sift_targ.detectAndCompute(img2,None)
    kp_img_temp = cv2.drawKeypoints(img1, kp_temp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('temp_keypoint.jpg',kp_img_temp)
    kp_img_targ = cv2.drawKeypoints(img2, kp_targ, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('targ_keypoint.jpg',kp_img_targ)
    neigh = NearestNeighbors(n_neighbors = 2, radius = 0.4)
    neigh.fit(des_targ)
    dis, point = neigh.kneighbors(des_temp, 2)
    df1 = pd.DataFrame(dis)
    df2 = pd.DataFrame(point)
    df = pd.concat([df1, df2], axis=1)
    df.columns = list(['d1', 'd2', 'x', 'y'])
    df['ratio'] = df['d1']/df['d2']
    dfkp = df[df['ratio'] < 0.75]
    dfkp = dfkp.drop_duplicates()
    x1_kp = pd.Series(kp_temp)[dfkp.index.values]
    x1_kp = x1_kp.values
    x2_kp = pd.Series(kp_targ)[dfkp['x']]
    x2_kp = x2_kp.values
    x1 = [x1_kp[idx].pt for idx in range(0, len(x1_kp))]
    x2 = [x2_kp[idx].pt for idx in range(0, len(x2_kp))]
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    inliners = 0
    most_inliners = -1
    for i in range(ransac_iter):
        all_index = list(range(x1.shape[0]))
        index = np.random.choice(range(x1.shape[0]), (1,3), replace=False)
        rand_x = [[x1[j], x2[j]] for j in index]
        A = np.matrix([[rand_x[0][0][0][0], rand_x[0][0][0][1], 1, 0, 0, 0], [0, 0, 0, rand_x[0][0][0][0], rand_x[0][0][0][1], 1], [rand_x[0][0][1][0], rand_x[0][0][1][1], 1, 0, 0, 0], [0, 0, 0, rand_x[0][0][1][0], rand_x[0][0][1][1], 1], [rand_x[0][0][2][0], rand_x[0][0][2][1], 1, 0, 0, 0], [0, 0, 0, rand_x[0][0][2][0], rand_x[0][0][2][1], 1]])
        B = np.matrix([[rand_x[0][1][0][0]], [rand_x[0][1][0][1]], [rand_x[0][1][1][0]], [rand_x[0][1][1][1]], [rand_x[0][1][2][0]], [rand_x[0][1][2][1]]])
        if np.linalg.matrix_rank(A) >= A.shape[0]:
            A_inv = np.linalg.inv(A)
            C = A_inv * B
            [all_index.remove(index[0][i]) for i in range(len(index[0]))]
            other_x1 = [x1[j] for j in all_index]
            other_x1 = np.matrix(other_x1)
            tran_mat = np.matrix([[C.item(0), C.item(1), C.item(2)], [C.item(3), C.item(4), C.item(5)], [0, 0, 1]])
            other_x1 = np.append(other_x1, np.ones((len(other_x1), 1)), axis=1)
            other_x2 = [x2[j] for j in all_index]
            other_x2 = np.matrix(other_x2)
            trans_x2 = np.linalg.multi_dot([tran_mat, other_x1.T]).T
            trans_x2 = np.delete(trans_x2, 2, axis=1)
            error = np.linalg.norm(other_x2 - trans_x2, axis=1)
            inliners = np.sum([0 if error_value > ransac_thr else 1 for error_value in error])
            if inliners > most_inliners:
                most_inliners = inliners
                final_tran_mat = tran_mat
    print('Number of inliners', most_inliners)
    A = final_tran_mat
    return A

def warp_image(img, A, output_size):
    # To do
    lim_H, lim_W = img.shape
    x = np.linspace(0, lim_W-1, lim_W)
    y = np.linspace(0, lim_H-1, lim_H)
    points = (y, x)
    rows, cols = output_size
    x_out = np.linspace(0, cols-1, cols)
    y_out = np.linspace(0, rows-1, rows)
    yy, xx = np.meshgrid(y_out,x_out)
    xy = np.hstack([xx.reshape(rows*cols, 1), yy.reshape(rows*cols, 1), np.ones((rows*cols, 1))])
    proj_coords = A @ xy.T
    proj_coords = np.delete(proj_coords, 2, 0)
    proj_coords = (proj_coords[1], proj_coords[0])
    img_warped = interpolate.interpn(points, img, proj_coords, method='linear')
    img_warped = img_warped.reshape(292, 452)
    img_warped = img_warped.T
    return img_warped

def align_image(template, target, A):
    # To do
    # Step 1 
    dum_mat = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    p0 = A - dum_mat
    p0 = p0.flatten()
    # Step 2
    temp_pad = np.ones((template.shape[0]+2, template.shape[1]+2)) * np.average(template)
    filter_x = np.reshape(np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]), -1)
    filter_y = np.reshape(np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]), -1)
    filter_x = np.reshape(np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]), -1)
    filter_y = np.reshape(np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]), -1)
    x_offset = 1
    y_offset = 1
    temp_pad[x_offset:template.shape[0]+x_offset,y_offset:template.shape[1]+y_offset] = template
    temp_filter_x = np.zeros(template.shape)
    temp_filter_y = np.zeros(template.shape)
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            b = np.reshape(temp_pad[i:i+3, j:j+3], -1)
            c_x = (filter_x.T @ b)
            c_y = (filter_y.T @ b)
            temp_filter_x[i][j] = c_x
            temp_filter_y[i][j] = c_y
    temp_filter_x = temp_filter_x.flatten()
    temp_filter_y = temp_filter_y.flatten()
    # Step 3
    dw_dp_u = np.tile(np.arange(template.shape[1]), (template.shape[0], 1))
    dw_dp_v = np.transpose([np.arange(template.shape[0])] * template.shape[1])
    dw_dp_1 = np.ones(template.shape)
    dw_dp_0 = np.zeros(template.shape)
    dw_dp_u = dw_dp_u.flatten()
    dw_dp_v = dw_dp_v.flatten()
    dw_dp_1 = dw_dp_1.flatten()
    dw_dp_0 = dw_dp_0.flatten()
    # Step 4
    temp_x_u = temp_filter_x * dw_dp_u
    temp_x_v = temp_filter_x * dw_dp_v
    temp_x_1 = temp_filter_x * dw_dp_1
    temp_y_u = temp_filter_y * dw_dp_u
    temp_y_v = temp_filter_y * dw_dp_v
    temp_y_1 = temp_filter_y * dw_dp_1
    temp_x_y_w_p = np.vstack((temp_x_u, temp_x_v, temp_x_1, temp_y_u, temp_y_v, temp_y_1))
    # Step 5
    H = temp_x_y_w_p @ temp_x_y_w_p.T
    inv_H = np.linalg.inv(H)
    # Step 6
    epsilon = 0.05
    delta_p_value = 100
    p_value = 100
    A_p = A
    template_flat = template.flatten()
    all_errors = []
    while delta_p_value > epsilon:
    # for i in range(300):
        # Step 7
        target_warp = warp_image(target, A_p, template.shape)
        target_warp = target_warp.flatten()
        # Step 8
        error_img = target_warp - template_flat
        error_value = np.linalg.norm(error_img, ord=2)
        all_errors.append(error_value)
        # Step 9
        F = temp_x_y_w_p @ error_img.T
        # Step 10
        delta_p = (inv_H @ F)
        delta_p_value = np.linalg.norm(delta_p)
        # Step 11
        A_p_delta = np.matrix([[delta_p[0] + 1, delta_p[1], delta_p[2]], [delta_p[3], delta_p[4] + 1, delta_p[5]], [0, 0, 1]])
        A_p =  A_p @ np.linalg.inv(A_p_delta)
        # print(delta_p_value)
    A_refined = A_p
    all_errors = np.array(all_errors)
    return A_refined, all_errors


def track_multi_frames(template, img_list):
    # To do
    ransac_thr = 10
    ransac_iter = 10000
    output_size = template.shape
    A_list = []
    # Comment everything after this
    x1, x2 = find_match(template, img_list[0])
    visualize_find_match(template, img_list[0], x1, x2)
    A_refined_old = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    # Till here
    for i in range(len(img_list)):
        print('Frame number', i+1)
        # x1, x2 = find_match(template, img_list[i])
        # visualize_find_match(template, img_list[i], x1, x2)
        # A_old = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
        A_refined, errors = align_image(template, img_list[i], A_refined_old)
        visualize_align_image(template, img_list[i], A_refined_old, A_refined, errors)
        template = warp_image(img_list[i], A_refined, output_size)
        template = np.uint8(template).copy()
        A_list.append(A_refined)
        A_refined_old = A_refined
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.savefig('SIFT_ratio_test.jpg', dpi=1000)
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)
    
    ransac_thr = 10
    ransac_iter = 1000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(np.abs(template-img_warped), cmap='jet')
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)
    
    # A_refined = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


