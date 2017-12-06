# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:44:04 2017

@author: hua
"""

# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
import numpy as np  
import matplotlib.pyplot as pt  
x = np.arange(0 , 360)  
y = np.sin( x * np.pi / 180.0)  
pt.plot(x,y)  
pt.xlim(0,360)  
pt.ylim(-1.2,1.2)  
pt.title("SIN function")  
pt.show()  