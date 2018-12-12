import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cv.namedWindow("w", cv.WINDOW_NORMAL)
points = np.array([(-100,-100,-100), (-100,100,-100),(100,-100,-100), (100,100,-100),(-100,-100,100), (-100,100,100),(100,-100,100), (100,100,100)]);
rotations = np.array([0,0,0])
translation = np.array([0,0,-30])
focal_length = 100;
center = (0,0)
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]],dtype=np.double)
dist_couffs = np.zeros((4,1))
rot_vec = np.array([(0,0,0)],dtype=np.double)
trans_vec = np.array([(0,0,-800)],dtype=np.double)
points_rotated = points
sign =0
while(sign!=ord('m')):
    img = np.zeros((400,400,3),np.uint8)
    if(sign==ord('w')):
        rot_vec = np.array([(0.01, 0,0)])
    elif sign==ord('e'):
        rot_vec =  np.array([(0, 0.01, 0)])
    elif sign==ord('r'):
        rot_vec = np.array([(0, 0, 0.01)])
    elif sign==ord('p'):
        trans_vec = trans_vec+np.array([(0,0,1)])

    rotated_mat, jaco = cv.Rodrigues(rot_vec)
    points_rotated = np.matmul(rotated_mat, points_rotated.T).T;     #points.T
    points_2d, hesi = cv.projectPoints(points_rotated, np.array([(0,0,0)],dtype=np.double),trans_vec,camera_matrix,dist_couffs)
    for point in points_2d:
        point_temp=(int(round(point[0,0]+200)), int(round(point[0, 1]+200)))
        cv.drawMarker(img,point_temp,(0,255,255))
    cv.imshow("w", img)
    sign = cv.waitKey(0)