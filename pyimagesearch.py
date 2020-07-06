#!/usr/bin/env python
# coding: utf-8

# In[4]:


## 7.1.3 A Basic Image Preprocessor
# import the necessary packages
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
    # store the target image width, height, and interpolation
    # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter 
    
    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)


# In[9]:


## 7.1.4 Building an Image Loader
# import the necessary packages
import numpy as np
import cv2
import os # to extract the names of subdirectories

class SimpleDatasetLoader:
    def __init__(self, preprocessors = None):
        #store the image preprocessor
        self.preprocessors = preprocessors
        
        #if the preprocessors are None, initialize them as an
        #empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verdose = -1):
        # initialize the list of features and labels
        data = []
        labels = []
        
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our hath has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
        
            #preprocessing
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
                
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(labels)
        
            # show an update every 'verdose' images
            if verdose > 0 and i > 0 and (i + 1) % verdose == 0:
                print("[INFO] processed {}/{}".format(i + 1,                     len(imagePaths)))

        # return a tuple  of the data and labels
        return(np.array(data), np.array(labels))
        


# In[11]:


## 7.2.3 Implementing k-NN
# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# In[ ]:





# In[ ]:




