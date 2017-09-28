import os
os.environ['THEANO_FLAGS']='device=cpu' 
os.environ['SIDEKIT']='libsvm=false,theano=false' 
import sys
import sidekit
import h5py
import numpy as np
import re

inpDirPath ='/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio'
ubmList,nameList = [],[]
for rootDirPath, subDir, files in os.walk(inpDirPath):
   for fileName in files:
       relFile = os.path.join(rootDirPath, fileName)
       ubmList.append(os.path.splitext(relFile)[0])
       nameList.append(os.path.splitext(relFile)[0].rsplit("/")[-1])
       
with open('../Text_DCASE/ubm_list.txt','w') as of:#'../Data/ubm_list.txt','w') as of:
   of.write("\n".join(ubmList))
with open('../Text_DCASE/name_list.txt','w') as of:
   of.write("\n".join(nameList))

	    


