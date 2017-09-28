import os
os.environ['THEANO_FLAGS']='device=cpu' 
os.environ['SIDEKIT']='libsvm=false,theano=false' 
import sys
import sidekit
import h5py
import logging
import numpy as np
import _pickle as cPickle

print('Load task definition')
'''
with open('/home/adit/Desktop/DCASE2017-baseline-system-master/Text_DCASE/ubm_list.txt') as inputFile:#'/home/deepu/research/DCASE2017/Data/ubm_list.txt') as inputFile:
    ubmList = inputFile.read().split('\n')


nameList = []
for a in ubmList:
    nameList.append(a.rsplit("/")[-1])
#directory = os.fsencode("/home/deepu/research/DCASE2017/Data/EmoDB_MFEC_HDF5")#"/home/adit/Desktop")# )
#os.chdir = "/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio/"
'''
with open('/home/adit/Desktop/DCASE2017-baseline-system-master/Text_DCASE/fold1_train_names.txt') as inputFile:#'/home/deepu/research/DCASE2017/Data/ubm_list.txt') as inputFile:
    ubmList = inputFile.read().split('\n')
nameList = []
for a in ubmList:
    nameList.append(a.rsplit(".")[0])


#for file in os.listdir(directory):
#	filename = os.fsdecode(file)
	#if filename.endswith(".wav"): 
	#print(os.path.join(directory, filename))
features_server = sidekit.FeaturesServer(features_extractor=None,
                                         feature_filename_structure="../HDF5_DCASE/Features/{}.h5",
                                         sources=None,
                                         dataset_list=["fb"],
                                         mask=None,
                                         feat_norm='cms',#'cmvn'
                                         global_cmvn=None,
                                         dct_pca=False,
                                         dct_pca_config=None,
                                         sdc=False,
                                         sdc_config=None,
                                         delta=False,
                                         double_delta=False,
                                         delta_filter=None,
                                         context=None,
                                         traps_dct_nb=None,
                                         rasta=False,
                                         keep_all_features=None)

#for i in range(0,11):	
print('Train the UBM by EM')
# Extract all features and train a GMM without writing to disk
distribNb = 512#np.power(2,i)
ubm = sidekit.Mixture()
p = "/home/adit/Desktop/"
print('__1__llk for '+str(distribNb))
llk = ubm.EM_split(features_server, nameList, distribNb, llk_gain=0.001, num_thread=3, save_partial=False)
    
print('__2__Saving for '+str(distribNb))
ubm.write(p+'/ubm_'+str(distribNb)+'.h5')
cPickle.dump( ubm, open( p+"models/ubm_"+str(distribNb)+".p", "wb" ) )
cPickle.dump( llk, open( p+"models/llk_"+str(distribNb)+".p", "wb" ) )

#ubm.write('/ubm_{}.h5'.format(distribNb))
#targetModelFile = '../Model/ubm'+str(distribNb)+'.h5'
#ubm.write(targetModelFile)

#enroll_idmap = sidekit.IdMap('/home/adit/Desktop/DCASE2017-baseline-system-master/451/ubm'+str(distribNb)+'.h5')
#es = sidekit.StatServer(distrib_nb=distribNb, feature_size=40)
#regulation_factor = 16  # MAP regulation factor
#enroll_sv = es.adapt_mean_map(ubm, regulation_factor)
#enroll_sv.write('/home/adit/Desktop/DCASE2017-baseline-system-master/451/adapted_gmm'+str(distribNb)+'.h5')
