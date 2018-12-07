import cv2
from collections import defaultdict
import pickle
import os
import numpy as np
import itertools
import DeleteBackGround
from sklearn.datasets import base





# def actualize_dict(dict, matches, desc):
# [dict[match]]

def getDescriptions1(path_folder1, name1, blur_mat, flag_resize = False):
    img = cv2.imread(path_folder1 + "/" + name1, 0)
    img = cv2.blur(img, (blur_mat,blur_mat))
    if flag_resize is not False:
        img = cv2.resize(img, (1653, 840))
    keys, des = descriptor.detectAndCompute(img, None)
    return img, keys, des
def getDescriptions2(path_folder1, name1, path_folder2, name2, blur):
    img1, keys1, des1 = getDescriptions1(path_folder1, name1, blur, True)
    img2, keys2, des2 = getDescriptions1(path_folder2, name2, blur, True)
    return img1, keys1, des1, img2, keys2, des2
def getDescriptions3(path_folder, blur):
    imageList = DeleteBackGround.loadFiles(path_folder)
    keys_list =[]
    des_list =[]
    img_list =[]
    for img_pat in imageList:
        img = cv2.imread(path_folder+"/"+img_pat,0)
        img=cv2.blur(img,(blur, blur))
        img = cv2.resize(img, (1653, 840))
        keys, des = descriptor.detectAndCompute(img, None)
        keys_list.append(keys)
        des_list.append(des)
        img_list.append(img)
    return img_list, keys_list, des_list
def bestMatches(keys_pattern, des_pattern, keys_trans, des_trans, amount=1):
    matcher = cv2.BFMatcher_create(cv2.cv2.NORM_HAMMING, crossCheck=False)
    matches_hier = matcher.knnMatch(des_pattern, des_trans, 2)
    matches = [i for (i, j) in matches_hier if i.distance < 0.8 * j.distance]
    matches = sorted(matches, key=lambda x: x.distance)
    points_patter = np.array([keys_pattern[num.queryIdx].pt for num in matches], dtype='float32').reshape(-1,1,2)
    points_trans = np.array([keys_trans[num.trainIdx].pt for num in matches], dtype='float32').reshape(-1,1,2)
    M, mask = cv2.findHomography(points_patter, points_trans, cv2.RANSAC)
    mask = mask.ravel().tolist()
    return M, mask, matches
def actualizeGraph(main_graph, temp_pairs):
    for pair in temp_pairs:
        key_first_node = None
        key_second_node = None
        for key in main_graph:
            if (np.isin(main_graph[key], pair[0])).all():
                key_first_node = key
            if (np.isin(main_graph[key], pair[1])).all():
                key_second_node = key
        if key_first_node is not None and key_second_node is not None:
            pass
        elif key_first_node is not None:
            main_graph[key_first_node].append(pair[1])
        elif key_second_node is not None:
            main_graph[key_second_node].append(pair[0])
        elif key_first_node is None and key_second_node is None:
            pass
    return main_graph
def make_part_of_dataset(des, dict, keys):
    data= []
    target = []
    target_names =[]
    #for key in keys:




cv2.namedWindow("draw keypoints", cv2.WINDOW_NORMAL)
cv2.namedWindow("draw keypoints2", cv2.WINDOW_NORMAL)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
descriptor = cv2.BRISK_create(thresh=30,octaves=3,patternScale=1.0)#,radiusList = np.ones((100)), numberList = np.ones((100))
path = "/home/krzysztof/Pulpit/banknoty/100zl_nobackground"
path_pattern_folder = "/home/krzysztof/Pulpit/banknoty"
name_pattern_img = "100zl_wzor.jpeg"

# images, keys, des = getDescriptions3(path, 7)
img_pattern, keys_pattern, des_pattern  = getDescriptions1(path_pattern_folder, name_pattern_img, 7)

# M, mask, matches = bestMatches(keys_pattern, des_pattern, keys, des)

imageList = DeleteBackGround.loadFiles(path)
img_matches =None
dict_matches  = defaultdict(list)



for  counter, imgPath in enumerate(imageList, start=0):
    img_trans, keys_trans, des_trans = getDescriptions1(path, imgPath, 9, True)
    M, mask, matches = bestMatches(keys_pattern, des_pattern, keys_trans, des_trans)
    matches_good = [matches[i] for i, elem in enumerate(mask) if elem is 1]
    for match in matches_good:
         dict_matches[keys_pattern[match.queryIdx].pt].append(des_trans[match.trainIdx])#

    # drawing keypoints
    img_matches = cv2.drawMatches(img_pattern, keys_pattern, img_trans,
                                  keys_trans, matches, img_matches, matchesMask=mask)#, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    cv2.imshow("draw keypoints", img_matches)
    cv2.waitKey(10)
    print(len(matches_good) , len(dict_matches))
    len_dict = [len(dict_matches[key]) for key in dict_matches]
    hist = np.histogram(len_dict, bins=range(1, 24))
    print(hist[0][:])

data = np.empty((0,64),np.uint8)
target = np.array([],np.uint8)
target_names = np.array([])

keys_good = [key for key in dict_matches if len(dict_matches[key])>8]
for counter, key in enumerate(keys_good, start=0):
    target_names = np.append(target_names, str(key))
    for row in dict_matches[key]:
        data = np.append(data, np.resize(row,(1, 64)), axis=0)
        target = np.append(target, counter)


data_base = base.Bunch(data= data, target=target, target_names=target_names)

# save database

f = open("/home/krzysztof/Pulpit/banknoty/100zl_des/database.base", 'wb')
pickle.dump(data_base, f)
f.close()





# combinations = itertools.combinations(imageList, 2)
# node_pairs = list()
# for i, comb in enumerate(combinations,start=0):
#     img1, keys1, des1, img2, keys2, des2 = getDescriptions2(path, comb[0], path, comb[1], 5)
#     M, mask, matches = bestMatches(keys1, des1, keys2, des2)
#     matches_good = [matches[i] for i, elem in enumerate(mask) if elem is 1]
#     [node_pairs.append((des1[match.queryIdx], des2[match.trainIdx])) for match in matches_good]
#     dict_matches=actualizeGraph(dict_matches, node_pairs)
#     len_dict = [len(dict_matches[key]) for key in dict_matches]
#     print(np.histogram(len_dict, bins=range(24))[0][12:])
#     img_matches = cv2.drawMatches(img1, keys1, img2,
#                                   keys2, matches, img_matches,
#                                   matchesMask=mask)  # , flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     cv2.imshow("draw keypoints", img_matches)
#     cv2.waitKey(0)