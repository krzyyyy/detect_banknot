import cv2
from collections import defaultdict
import pickle
import os
import numpy as np
import itertools
import DeleteBackGround
from sklearn.datasets import load_iris
import sklearn as skl


iris =load_iris()
# print(iris.keys())
# print(iris['data'])
# print(iris['target'])
# print(iris['target_names'])
new_dataset = skl.datasets.base.Bunch(target=iris['target'], data=iris['data'], target_names=iris['target_names'])

