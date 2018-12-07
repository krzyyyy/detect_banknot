import cv2
import pickle
import os
import numpy as np
import itertools


def loadFiles(path):
    os.chdir(path)
    filesList=os.listdir('.')
    clearFilesList = [i for i in filesList if i[-4:]=='.jpg' or i[-5:]=='.jpeg' or i[-4:]=='.png']
    print(clearFilesList)
    return  clearFilesList
def deleteBackGround(img):
    deleteBackGround.counter+=1
    dest_path = "/home/krzysztof/Pulpit/banknoty/100zl_nobackground"
    cv2.namedWindow("okno", cv2.WINDOW_NORMAL)
    cv2.namedWindow("okno2", cv2.WINDOW_NORMAL)
    sign = -1
    thres1, thres2, size = 25, 35, 3
    dst4 = []
    while (True):
        dst2 = img.copy()
        dst2 = cv2.medianBlur(dst2,21)
        dst3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        imgp = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        if sign== ord('q'):
            break
        elif sign== ord('p'):
            thres1=thres1+1
        elif sign==ord('m'):
            thres1 = thres1-1
        elif sign== ord('o'):
            thres2=thres2+2
        elif sign==ord('n'):
            thres2 = thres2-2
        elif sign==ord('s'):
            size=size+2
        elif sign==ord('w'):
            cv2.imwrite(dest_path + "/" + deleteBackGround.counter.__str__() + ".png", dst4)
        print(f"threshold1: {thres1} \n threshold2: {thres2}")
        # thresholding image
        cimg = cv2.adaptiveThreshold(dst2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,thres2,thres1/10)
        # looking for contours
        cimg, contours, hierarhy = cv2.findContours(cimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # finding the biggest contour
        max_region = max(contours, key=lambda item: cv2.contourArea(item))

        # fitting the smallest rotated rectangle
        maxrect = cv2.minAreaRect(max_region)
        maxbox=cv2.boxPoints(maxrect)
        maxbox = np.int0(maxbox)
        # expanding dimention of binary image
        cimg = np.expand_dims(cimg,axis=2)
        cimg3d = np.concatenate((cimg,np.zeros((img.shape[0], img.shape[1], 2),np.uint8)),axis=2)
        # drawing conturs
        cv2.drawContours(cimg3d,[maxbox],-1,(240,20,230),3)
        cv2.drawContours(cimg3d, [max_region],-1,(128,240,60),3)

        # filling the biggest contour
        convex_hull=cv2.convexHull(max_region)
        dst3= cv2.fillConvexPoly(dst3,convex_hull,(255))
        # masking a orginal image
        dst4 = cv2.bitwise_and(img, dst3)
        #cutting the orginal image
        x,y, w, h=cv2.boundingRect(convex_hull)
        dst4 = dst4[ y:y+h, x:x+w]
        cv2.imshow("okno", dst4)
        cv2.imshow("okno2", cimg3d)
        sign = cv2.waitKey(10)
        break
    cv2.destroyAllWindows()
    return dst4
deleteBackGround.counter=0

def perspectivWarp(img_pattern, img_transformed):
    cv2.namedWindow("window_1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("window_2", cv2.WINDOW_NORMAL)
    img_matches = None
    descriptor = cv2.BRISK_create(thresh=30,octaves=9)
    matcher = cv2.BFMatcher_create(cv2.cv2.NORM_HAMMING,crossCheck=True)
    keys_pattern, des_pattern = descriptor.detectAndCompute(img_pattern,None)
    keys_trans, des_trans = descriptor.detectAndCompute(img_transformed, None)
    matches = matcher.match(des_pattern, des_trans)
    matches = sorted(matches, key = lambda x: x.distance)
    img_matches = cv2.drawMatches(img_pattern, keys_pattern , img_transformed, keys_trans,  matches[:10], img_matches,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS )

    pattern_points = np.array([keys_pattern[num.queryIdx].pt for num in matches[:20]], dtype='float32').reshape(-1,1,2)
    trans_points = np.array([keys_trans[num.trainIdx].pt for num in matches[:20]], dtype='float32').reshape(-1,1,2)
    M, mask = cv2.findHomography(pattern_points, trans_points, cv2.RANSAC)
    M2, mask2 = cv2.findHomography(trans_points, pattern_points, cv2.RANSAC)
    matchesMask = mask.ravel().tolist()

    heigh, width = img_pattern.shape
    rect = np.array([[0, 0], [0, heigh - 1], [width - 1, heigh - 1], [width - 1, 0]], np.float32)
    rect = np.expand_dims(rect, axis=1)
    dst_points = cv2.perspectiveTransform(rect, M)
    dst_points = dst_points[:, 0, :]
    cv2.polylines(img_transformed, np.int32([dst_points]),True, (255))

    dst = cv2.warpPerspective(img_transformed, M2, (img_pattern.shape[1],img_pattern.shape[0]))


    cv2.imshow("window_1", dst)
    cv2.imshow("window_2", img_transformed)
    cv2.waitKey(10)
    return np.int32([dst_points])


# def getCombinations(keys_pattern, keys_trans, matches):
#     combinations = itertools.combinations(matches , 4)
#     matrixcom = [];
#     for com in combinations:
#         patternpoints = np.array([keys_pattern[com[num].queryIdx].pt for num in range(4)],dtype='float32')
#         transpoints = np.array([keys_trans[com[num].trainIdx].pt for num in range(4)], dtype= 'float32')
#         mat = cv2.getPerspectiveTransform(patternpoints,transpoints);
#         matrixcom.append(mat)
#     dets = [np.linalg.det(mat) for mat in matrixcom]
#     print(dets, '\n')
#     return matrixcom

# paths to images and loading pattern image
def main():
    path = "/home/krzysztof/Pulpit/banknoty/100zl"
    path_pattern_img = "/home/krzysztof/Pulpit/banknoty/100zl_wzor.jpeg"
    img_pattern = cv2.imread(path_pattern_img, 0)
    imageList = loadFiles(path)

    for  counter, imgPath in enumerate(imageList, start=0):
        img = cv2.imread(path + "/" + imgPath, 0)
        des = deleteBackGround(img)
        des2 = perspectivWarp(img_pattern, img)


