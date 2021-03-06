import os
os.environ['THEANO_FLAGS']='device=cpu' 
os.environ['SIDEKIT']='libsvm=false,theano=false' 
import sys
import sidekit
import h5py
import logging
import numpy as np


print('Load task definition')
with open('/home/adit/Desktop/DCASE2017-baseline-system-master/Text_DCASE/fold1_train_names.txt') as inputFile: 
    ubmList = inputFile.read().split('\n')
nameList = []
for a in ubmList:
    nameList.append(a.rsplit('.')[0])

directory = '/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio/'
extractor = sidekit.FeaturesExtractor(audio_filename_structure = "/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio/{}.wav",
					feature_filename_structure = "/home/adit/Desktop/shikha_maam/Features/{}.h5",
					sampling_frequency  =   16000,
                                        lower_frequency     =   10,
                                        higher_frequency    =   8000,
                                        filter_bank         =   "log",
                                        filter_bank_size    =   40,
                                        window_size         =   0.040,
                                        ceps_number         =   12,
                                        shift               =   0.020,
                                        vad                 =   None,
                                        snr                 =   None,
                                        pre_emphasis        =   0.97,
                                        save_param          =   ["fb"],
                                        keep_all_features   =   None)

show_list = np.unique([nameList])
channel_list = np.zeros_like(show_list, dtype = int)

logging.info("Extract features and save to disk")
extractor.save_list(show_list = show_list, channel_list = channel_list)

