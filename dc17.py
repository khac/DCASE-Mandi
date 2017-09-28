import os
os.environ['THEANO_FLAGS']='device=cpu' 
os.environ['SIDEKIT']='libsvm=false,theano=false' 
import sidekit
import h5py
import numpy as np



directory = os.fsencode("/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio")#"/home/adit/Desktop")# )
#os.chdir = "/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio/"

for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.endswith(".wav"): 
	# print(os.path.join(directory, filename))
		
		extractor = sidekit.FeaturesExtractor(audio_filename_structure = "/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio/"+filename,
												feature_filename_structure = "/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio_mfec_hdf5/"+filename+".h5",
												sampling_frequency= 44100,
												lower_frequency=20,
												higher_frequency = 22000,
												filter_bank = "log",
												filter_bank_size = 40,
												window_size = 0.040,
												ceps_number = 12,
												shift = 0.020,
												vad = "snr",
												snr = 40,
												pre_emphasis= 0.97,
												save_param = ["fb"],
												keep_all_features = True)
		extractor.save("")
		
		file = h5py.File('/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio_mfec_hdf5/'+filename+'.h5','r+')
		data = file['/fb']
		#print(data)
		tocsv = data[:,:]
		data1 = file['/fb_mean']
		tomean = data1[:,]
		tomean = tocsv-tomean
		#print(tomean,tocsv-tomean)
		np.savetxt('/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio_mfec_txt/'+filename+'.txt', tocsv, delimiter=',', newline='\n')  
		np.savetxt('/home/adit/Desktop/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/audio_mfec_txt_mean/'+filename+'.txt', tomean, delimiter=',', newline='\n')

	else:
		continue

