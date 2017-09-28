import os
os.environ['THEANO_FLAGS']='device=cpu' 
os.environ['SIDEKIT']='libsvm=false,theano=false' 
import sys
import sidekit
import h5py
import logging
import numpy as np

models,segments = [],[]
with open('/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/evaluation_setup/fold1_train.txt','r+') as op:
   for i in op:
       i=i.replace('audio/','').replace('.wav','')
       i = i.rstrip('\n').split('\t')
       models.append(i[1])
       segments.append(i[0])


print(len(models))
print(len(segments))

enroll_idmap = sidekit.IdMap()
enroll_idmap.leftids = np.asarray(models)
enroll_idmap.rightids = np.asarray(segments)
enroll_idmap.start = np.empty(enroll_idmap.rightids.shape, dtype='|O')
enroll_idmap.stop = np.empty(enroll_idmap.rightids.shape, dtype='|O')
enroll_idmap.validate()
enroll_idmap.write('/home/adit/Desktop/enroll_ids1.h5')#DCASE2017-baseline-system-master/HDF5_DCASE/enroll_ids4.h5')


models, segments,targets = [],[],[]

with open('/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/evaluation_setup/fold1_evaluate.txt','r+') as op:
   for i in op:
       i=i.replace('audio/','').replace('.wav','')
       i = i.rstrip('\n').split('\t')
       models.append(i[1])
       segments.append(i[0])
       targets.append('target')

print(len(models))
print(len(segments))

key = sidekit.Key(models=np.array(models), testsegs=np.array(segments), trials=np.array(targets))
key.write('/home/adit/Desktop/key1.h5')

ndx = key.to_ndx()
ndx.write('/home/adit/Desktop/ndx1.h5')

