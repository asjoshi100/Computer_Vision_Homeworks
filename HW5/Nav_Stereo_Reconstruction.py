import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
from scipy.linalg import svd

import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy.linalg import matrix_rank

#def find_match(img1, img2):
#    # TO DO
#    return pts1, pts2


def find_match(img1, img2):
    # To do
    #gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img1,None)
    
    #img=cv2.drawKeypoints(img1,kp,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('sift_keypoints.jpg',img)
    
    #gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(img2,None)
    
    #img=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('sift_keypoints2.jpg',img2)
    
    
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    
    targ_kp = neigh.kneighbors(des,2,return_distance = True)
    
    df1 = pd.DataFrame(targ_kp[0])
    df2 = pd.DataFrame(targ_kp[1])
    df =pd.concat([df1, df2], axis=1)
    df.columns = list(['d1','d2','x','y'])
    df['ratio'] = df['d1']/df['d2']
    
    dfkp = df[df['ratio']<.71]

    
    
    x1n = pd.Series(kp)[dfkp.index.values]
    x1n = x1n.values
    
    x2n = pd.Series(kp2)[dfkp['x']]
    x2n = x2n.values
    
    x1= [x1n[idx].pt for idx in range(0, len(x1n))]
    x2 = [x2n[idx].pt for idx in range(0, len(x2n))]
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2

def compute_F(pts1, pts2):
    # TO DO
    ransac_iter=100000;
    inliers=np.zeros((ransac_iter,1));
    Ftemp= [];
    n=0;
    ransac_threshold=0.01;
    rn = len(pts1)
    import random
    while(n<ransac_iter):
        #Generate 5 random numbers between 10 and 30
     count= 0
     A = []
     randomlist = random.sample(range(0, rn), 8)
     for i in randomlist:
      Atemp = [pts1[i,0]*pts2[i,0],pts1[i,1]*pts2[i,0],pts2[i,0],pts1[i,0]*pts2[i,1],pts1[i,1]*pts2[i,1],pts2[i,1],pts1[i,0],pts1[i,1],1.0]
      A.append(Atemp) 
     A = np.array(A)
     fn = null_space(A)[:,0]
     
     f = fn.reshape((3,3))
     f = f/f[2,2]
     [U,D,V] = svd(f)
     D[2] = 0
     f_mod = U@np.diag(D)@V
     for i in range(0,rn):
        u = np.array([pts1[i,0],pts1[i,1],1])
        v = np.array([pts2[i,0],pts2[i,1],1])
        err = (v.T)@f_mod@u
       
        if(np.abs(err)<ransac_threshold):
            count = count + 1
     inliers[n] = count
     Ftemp.append(f_mod)
     n = n+1
    idx = np.argmax(inliers)
    F = Ftemp[idx]
                
    return F


def triangulation(P1, P2, pts1, pts2):
    # TO DO
    n = len(pts1)
    pts3D = []
    
    for i in range(0,n):
        Au = [[0,-1,pts1[i,1]],[1,0,-pts1[i,0]],[-pts1[i,1],pts1[i,0],0]]@P1
        Ap = [[0,-1,pts2[i,1]],[1,0,-pts2[i,0]],[-pts2[i,1],pts2[i,0],0]]@P2
        Aaug = np.vstack([Au,Ap])
        [U,S,V] = svd(Aaug)
        X_temp = V[0:3,3]/V[3,3]
        pts3D.append(X_temp)
#        ns = null_space(Aaug)
#        print(ns.shape)
#        X_temp = [ns[0],ns[1],ns[2]]/ns[3]
        
        
        
    pts3D = np.array(pts3D)
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    L = []

    for i in range(0,4):
        R = Rs[i]
        C = Cs[i]
        X = pts3Ds[i]
        count = 0
        for j in range(0,len(X)):
            c = R[2,:]@((X[j,:]-C.T).T)
            if(c>0):
                count = count +1
        L.append(count)
        
    idx = np.argmax(L)
    R = Rs[idx]
    C = Cs[idx]
    pts3D = pts3Ds[idx]
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    rx = C/np.linalg.norm(C)
    rzt = np.array([0,0,1]).T
    rz = ((rzt - (rzt.T@rx)*rx.T)/np.linalg.norm((rzt-(rzt.T@rx))*rx))
    ry = np.cross(rz,rx.T)
    
    Rr = np.array([rx.T,ry,rz]).reshape((3,3))
    
    H1=K@Rr@np.linalg.inv(K);
    H2=K@Rr@(R.T)@np.linalg.inv(K);
    return H1, H2

def convert_pts_to_keypoints(pts, size=5): 
    kps = []
    if pts is not None: 
        if pts.ndim > 2:
            # convert matrix [Nx1x2] of pts into list of keypoints  
            kps = [ cv2.KeyPoint(p[0][0], p[0][1], _size=size) for p in pts ]          
        else: 
            # convert matrix [Nx2] of pts into list of keypoints  
            kps = [ cv2.KeyPoint(p[0], p[1], _size=size) for p in pts ]                      
    return kps      
def dense_match(img1, img2):
    # TO DO
    sift = cv2.xfeatures2d.SIFT_create()
    #img1 = np.pad(img1,(100,100))
    #img2 = np.pad(img2,(100,100))

    s = img1.shape[1]
    X,Y = np.meshgrid(range(img1.shape[1]),range(img1.shape[0]),indexing='ij')
    cg = np.array([X,Y]).T
    #kps = convert_pts_to_keypoints(cg, size=1)
    kps = []
    for i in range(np.shape(cg)[0]):
     for j in range(np.shape(cg)[1]):
        k = cv2.KeyPoint(cg[i][j][0], cg[i][j][1], _size=5)
        kps.append(k)
        
    kp1,des1 = sift.compute(img1,kps)
    imk1 = np.zeros((img1.shape[0],img1.shape[1],128))   
    for i in range(img1.shape[0]):
         for j in range(img1.shape[1]):
             imk1[int(kp1[i*s + j].pt[1]),int(kp1[i*s + j].pt[0])] = des1[i*s + j]
             #imk1[i,j]  = des1[i*img1.shape[1] + j]
    kp2,des2 = sift.compute(img2,kps)
    imk2 = np.zeros((img1.shape[0],img1.shape[1],128))   
    for i in range(img1.shape[0]):
         for j in range(img1.shape[1]):
             imk2[int(kp2[i*s + j].pt[1]),int(kp2[i*s + j].pt[0])] = des2[i*s + j]
             #imk2[i,j] = des2[i*img1.shape[1]+ j]
    disparity = np.zeros((img1.shape[0],img1.shape[1]))
    for i in range(img1.shape[0]):
         for j in range(img1.shape[1]):
             ref = imk1[i,j].reshape((1,128))
             temp = imk2[i,0:j+1]
             
             vnorm = np.linalg.norm(temp-ref,axis = 1)
             #l = np.argmin(vnorm)
             l = np.where(vnorm == vnorm.min())[0]
             
             disparity[i][j] = np.abs(l-j).min()
    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    print('Start find match')
    pts1, pts2 = find_match(img_left, img_right)
    print('finish find match')
    visualize_find_match(img_left, img_right, pts1, pts2)
    print('Finish visualize find match')

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
