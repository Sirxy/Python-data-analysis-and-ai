#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import cv2


# In[2]:


# 人脸，特征数据的，获取额人脸特征
# 交给cv2的算法，算法就可以根据特征查找人脸


# In[5]:


huazai = cv2.imread("./huazai.jpg")

# 声明算法
face_detect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


# In[6]:


face_zone = face_detect.detectMultiScale(huazai, scaleFactor=1.1, minNeighbors=5)


# In[7]:


print(face_zone)

cv2.imshow('star', huazai)

cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




