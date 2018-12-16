import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cv.namedWindow("w", cv.WINDOW_NORMAL)
img_points = np.empty((0,4), np.uint8)
rang = [-100, 100]
points = np.array([(-100,-100,-100), (-100,100,-100),(100,-100,-100), (100,100,-100),(-100,-100,100), (-100,100,100),(100,-100,100), (100,100,100)]);
for r in rang:
    img_points = np.append(img_points,np.where(points[:,0]==r),axis=0)
    img_points = np.append(img_points, np.where(points[:, 1] == r), axis=0)
    img_points = np.append(img_points, np.where(points[:, 2] == r), axis=0)


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
    index = np.where(points_rotated[:,2]==min(points_rotated,key=lambda x: x[2])[2])
    img_mean_points = np.array([[np.mean(points_rotated[x],axis=0) for x in img_points]])
    points_2d, hesi = cv.projectPoints(points_rotated, np.array([(0,0,0)],dtype=np.double),trans_vec,camera_matrix,dist_couffs)
    points_mean_2d, hesi = cv.projectPoints(img_mean_points,np.array([(0., 0., 0.)], dtype=np.double),trans_vec,camera_matrix, dist_couffs)

    tab = np.array([x for x in range(6)], np.uint8)
    tab_sort = sorted(tab, key=lambda x: img_mean_points[0][x][2])
    for count, id in enumerate(tab_sort):
        if_in = False


    for point in points_2d:
        point_temp=(int(round(point[0,0]+200)), int(round(point[0, 1]+200)))
        cv.drawMarker(img,point_temp,(0,255,255))
    for point in points_mean_2d:
        point_temp = (int(round(point[0, 0] + 200)), int(round(point[0, 1] + 200)))
        cv.drawMarker(img, point_temp, (255, 255, 0))
    point_temp = (int(round(points_2d[index[0][0],0, 0] + 200)), int(round(points_2d[index[0][0],0, 1] + 200)))
    cv.drawMarker(img, point_temp, (255, 0, 255))
    cv.imshow("w", img)
    sign = cv.waitKey(0)
    print(min(points_rotated,key=lambda x: x[2]),"\n---------------------------------------")
    print(points_rotated)
    print(index)