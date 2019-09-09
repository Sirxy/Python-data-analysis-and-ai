#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import cv2


# In[2]:


# 加载算法
face_detect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


# In[3]:


cap = cv2.VideoCapture("./mp4/陶西当众花式表白安谧老师，这画面真的太甜了！.mp4") 


# In[4]:


cap.get(propId=cv2.CAP_PROP_FPS)  # 一秒多少帧


# In[5]:


while True:
    flag, frame = cap.read()
    
    if flag == False:
        break
    
    gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
    
    face_zone = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    for x,y,w,h in face_zone:
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+w, y+h), color=[0,0,255], thickness=2)
        cv2.circle(frame, center=(x+w//2,y+h//2), radius=w//2, color=[0,255,0], thickness=2)
    
    cv2.imshow("video", frame)
    
    if ord("q") == cv2.waitKey(18):
        break
cv2.destroyAllWindows()
cap.release()


# In[ ]:





# In[ ]:




