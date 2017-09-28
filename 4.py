import os
os.environ['THEANO_FLAGS']='device=cpu' 
os.environ['SIDEKIT']='libsvm=false,theano=false' 
import sys
import sidekit
import h5py
import logging
import numpy as np

directory = os.fsencode("/home/adit/Desktop/DCASE2017-baseline-system-master/Model_DCASE")#"/home/adit/Desktop")
distribNb = 2048
ubm = sidekit.Mixture()
enroll_stat = sidekit.StatServer(distrib_nb=distribNb, feature_size=40)
regulation_factor = 3  # MAP regulation factor
enroll_sv = enroll_stat.adapt_mean_map(ubm, regulation_factor)
enroll_sv.write('gmm_adapted.h5')

