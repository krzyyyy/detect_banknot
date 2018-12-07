import cv2
import pickle
import os
import numpy as np

def loadFiles(path):
    os.chdir(path)
    filesList=os.listdir('.')
    clearFilesList = [i for i in filesList if i[-4:]=='.jpg' or i[-5:]=='.jpeg']
    print(clearFilesList)
    return  clearFilesList
def deleteBackGround(img):
    while (True):
        dst2 = img.copy()
        dst2 = cv2.medianBlur(dst2,21)
        dst = np.zeros((img.shape[0], img.shape[1],3),  np.uint8)
        dst3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        if cv2.waitKey(30)== ord('q'):
            break
        elif  cv2.waitKey(30)== ord('p'):
            thres1=thres1+1
        elif cv2.waitKey(33)==ord('m'):
            thres1 = thres1-1
        elif  cv2.waitKey(30)== ord('o'):
            thres2=thres2+2
        elif cv2.waitKey(30)==ord('n'):
            thres2 = thres2-2
        elif cv2.waitKey(30)==ord('s'):
            size=size+2
        print(f"threshold1: {thres1} \n threshold2: {thres2}")
        cimg = cv2.adaptiveThreshold(dst2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,thres2,thres1/10)
        cimg, contours, hierarhy = cv2.findContours(cimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        max_region = max(contours, key=lambda item: cv2.contourArea(item))
        # maxrect = cv2.minAreaRect(max_region)
        # maxbox=cv2.boxPoints(maxrect)
        # maxbox = np.int0(maxbox)
        # cv2.drawContours(dst,[maxbox],-1,(240,20,230),3)
        # max_region2 = cv2.approxPolyDP(max_region, 2, True)
        convex_hull=cv2.convexHull(max_region)
        # dst=cv2.fillPoly(dst,max_region2,(255,255,255))
        # cv2.drawContours(dst,[max_region],-1,(255,255,0),3)
        dst3= cv2.fillConvexPoly(dst3,convex_hull,(255))
        dst4 = cv2.bitwise_and(img, dst3)
        # dst3 = cv2.drawContours(dst3, [convex_hull],-1,(20,230,10),3)
        # M = cv2.getRotationMatrix2D(maxrect[0],maxrect[2], 1)
        # rows, cols = img.shape;
        # dst3=cv2.warpAffine(dst,M,(rows, cols))
    return dst4


datebase = []
descriptor = cv2.BRISK_create()
path = "/home/krzysztof/Pulpit/banknoty/100zl"
dest_path = "/home/krzysztof/Pulpit/banknoty/100zl_nobackground"
# os.mkdir("/home/krzysztof/Pulpit/banknoty/100zl_nobackground")
imageList = loadFiles(path)
thres1, thres2, size = 25, 35, 3
cv2.namedWindow("okno",cv2.WINDOW_NORMAL)
cv2.namedWindow("okno2",cv2.WINDOW_NORMAL)
cv2.namedWindow("okno3",cv2.WINDOW_NORMAL)
for  counter, imgPath in enumerate(imageList, start=0):
    img = cv2.imread(path+"/"+imgPath, 0)
    print(counter)
    while (True):
        dst2 = img.copy()
        # kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        dst2 = cv2.medianBlur(dst2,21)
        # dst2 = cv2.filter2D(dst2,-1,kernel_sharp)
        dst = np.zeros((img.shape[0], img.shape[1],3),  np.uint8)
        dst3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        if cv2.waitKey(30)== ord('q'):
            break
        elif  cv2.waitKey(30)== ord('p'):
            thres1=thres1+1
        elif cv2.waitKey(33)==ord('m'):
            thres1 = thres1-1
        elif  cv2.waitKey(30)== ord('o'):
            thres2=thres2+2
        elif cv2.waitKey(30)==ord('n'):
            thres2 = thres2-2
        elif cv2.waitKey(30)==ord('s'):
            size=size+2
        print(f"threshold1: {thres1} \n threshold2: {thres2}")
        cimg = cv2.adaptiveThreshold(dst2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,thres2,thres1/10)
        cimg, contours, hierarhy = cv2.findContours(cimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        dst = cv2.drawContours(dst,contours,-1,(20,30,240),3)

        max_region = max(contours, key=lambda item: cv2.contourArea(item))
        maxrect = cv2.minAreaRect(max_region)
        maxbox=cv2.boxPoints(maxrect)
        maxbox = np.int0(maxbox)
        cv2.drawContours(dst,[maxbox],-1,(240,20,230),3)
        max_region2 = cv2.approxPolyDP(max_region, 2, True)
        convex_hull=cv2.convexHull(max_region)
        dst=cv2.fillPoly(dst,max_region2,(255,255,255))
        cv2.drawContours(dst,[max_region],-1,(255,255,0),3)
        dst3= cv2.fillConvexPoly(dst3,convex_hull,(255))
        dst4 = cv2.bitwise_and(img, dst3)
        # dst3 = cv2.drawContours(dst3, [convex_hull],-1,(20,230,10),3)
        # M = cv2.getRotationMatrix2D(maxrect[0],maxrect[2], 1)
        # rows, cols = img.shape;
        # dst3=cv2.warpAffine(dst,M,(rows, cols))
        cv2.imshow("okno", dst4)
        cv2.imshow("okno2",dst)
        cv2.imshow("okno3",dst3)
    cv2.imwrite(dest_path+"/"+counter.__str__()+".png",dst4)
print("wymiary:  ")



#f = open("/home/krzysztof/Pulpit/banknoty/100zl_des/data", 'wb')
#pickle.dump(datebase, f)
#f.close()
