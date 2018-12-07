import cv2
import numpy as np
import statistics as stat
import scipy
import matplotlib.pyplot as plt
import pickle
from os import *


def perspectivWarp(img_pattern, img_transformed):
    perspectivWarp.counter += 1
    img_matches = None
    descriptor = cv2.BRISK_create(thresh=30,octaves=3)
    keys_pattern, des_pattern = descriptor.detectAndCompute(img_pattern,None)
    keys_trans, des_trans = descriptor.detectAndCompute(img_transformed, None)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    # PIERWSZY SPOSÓB -------------------------------------------
    matches = matcher.knnMatch(des_pattern, des_trans, 2)
    matches2 =[]
    [matches2.append(i) for i, j in matches if i.distance< 0.7*j.distance]
    matches2 = sorted(matches2, key= lambda x: x.distance)
    img_matches = cv2.drawMatches(img_pattern, keys_pattern, img_transformed, keys_trans, matches2,img_matches,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    means = np.array([mat.distance for mat in matches2[:10]])
    check = [np.mean(means), np.median(means)]
    print(check)
    if len(matches2) < 10: return np.int32([]), img_matches


    pattern_points = np.array([keys_pattern[num.queryIdx].pt for num in matches2], dtype='float32').reshape(-1,1,2)
    trans_points = np.array([keys_trans[num.trainIdx].pt for num in matches2], dtype='float32').reshape(-1,1,2)

    if check[0] >85 and check[1]>85:
        return np.int32([]), img_matches
    M, mask = cv2.findHomography(pattern_points, trans_points, cv2.RANSAC)
    if M is None:
        return np.int32([]), img_matches

    cv2.putText(img_matches, np.sum(mask).__str__(), (img_matches.shape[1] - 40, img_matches.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 230), 3)
    M2, mask2 = cv2.findHomography(trans_points, pattern_points, cv2.RANSAC)

    heigh, width = img_pattern.shape
    rect = np.array([[0, 0], [0, heigh - 1], [width - 1, heigh - 1], [width - 1, 0]], np.float32)
    rect = np.expand_dims(rect, axis=1)
    dst_points = cv2.perspectiveTransform(rect, M)
    dst_points = dst_points[:, 0, :]
    cv2.polylines(img_transformed, np.int32([dst_points]),True, (255))

    dst = cv2.warpPerspective(img_transformed, M2, (img_pattern.shape[1],img_pattern.shape[0]))
    return np.int32(dst_points), img_matches
perspectivWarp.counter = 1

cv2.namedWindow("window_1", cv2.WINDOW_NORMAL)
cv2.namedWindow("window_2", cv2.WINDOW_NORMAL)
cap =cv2.VideoCapture(0)
path_pattern_img = "/home/krzysztof/Pulpit/banknoty/100zl_wzor.jpeg"
img = cv2.imread(path_pattern_img, 0)
img = cv2.resize(img,None,fx=0.25,fy=0.25)
img_matches = None
while(True):
    ret, frame = cap.read()
    points, img_matches = perspectivWarp(img, frame)
    if len(points) == 0:
        cv2.putText(frame,"No banknot", (10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,230),3)
    else:
        cv2.putText(frame, "100 zl", tuple(points[0,:]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,230),3)
        cv2.polylines(frame, [points], True, (0, 0, 255))
    cv2.imshow("window_1", frame)
    cv2.imshow("window_2", img_matches)
    cv2.waitKey(10)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break




