import numpy as np
import random
import sys
import glob
import librosa
sys.path.append('../')
import time

# ignore librosa errors
import warnings
warnings.filterwarnings('ignore')

class SlakhToNpz():
    def __init__(self, inst_num, train_npz_num, valid_npz_num):
        self.inst_num = inst_num
        self.valid_npz_num = valid_npz_num
        self.train_npz_num = train_npz_num
        self.sec = 10
        self.fs = 44100
        self.target_fs = 16000
        self.slakh_folder_path = '../data/slakh2100_flac/'
        self.save_folder_path ='../data/slakh_inst{0}/'.format(inst_num)
        
        self.valid_subfolders = glob.glob(self.slakh_folder_path + 'validation/*')
        self.test_subfolders = glob.glob(self.slakh_folder_path + 'test/*')
        self.debug = None
        self.debug2 = None
    def _read_as_mono(self, filename):
        data, _ = librosa.load(filename,  sr=self.fs, dtype=np.float32, mono=True)
        return data
        
    def _downsampilng(self, data):
        y = librosa.resample(data, self.fs, self.target_fs)
        return y
        
    def _to_npz(self, file_path, mixture, sources):
        np.savez(file_path, mixture=mixture, sources=sources, instruments_num=self.inst_num)
        
    def _silent_exist(self, x):
        norm = np.linalg.norm(x, ord=1, axis=1)
        return  True if np.any(norm == 0) else False
    
    def _random_cutting(self, sources):
        print('sources shape:', sources.shape)
        sources_num = sources.shape[0]
        source_len = sources.shape[1]
        cut_sources_array = np.zeros((sources_num, self.fs*self.sec))
        offset = random.randrange(source_len - self.fs*self.sec)
        cut_sources_array[:, :] = sources[:, offset:offset+self.fs*self.sec]
        return cut_sources_array
    
    def _is_unique_shape(self, source_list):
        return len(set(source_list)) == 1
    
    def make_train_tracks(self, mode='train'):
        assert mode == 'train' or mode == 'validation', 'InvalidArgument'
        subfolders = glob.glob(self.slakh_folder_path + '{0}/*'.format(mode))
        npz_num = self.train_npz_num if mode == 'train' else self.valid_npz_num
        num_npz_per_track = npz_num / len(subfolders) + 1
        
        npz_idx = 0
        for track_path in subfolders:
            print('track_path:', track_path)
            sources = glob.glob(track_path + '/stems/*')
            
            sources_list = []
            sources_shape = []
            for i, source_path in enumerate(sources):
                data = self._read_as_mono(source_path)
                sources_list.append(data)
                sources_shape.append(data.shape)
                
            if not self._is_unique_shape(sources_shape):
                continue
                             
            npz_num = 0
            while npz_num < num_npz_per_track:
                start = time.time()
                self.debug  = sources_list
                sample_sources = np.array(random.sample(sources_list, self.inst_num))
                self.debug2 = sample_sources
                cut_sources = self._random_cutting(sample_sources)
                
                ds_cut_sources = np.zeros((self.inst_num, self.target_fs*self.sec))
                for i, cut_source in enumerate(cut_sources):    
                    ds_cut_source = self._downsampilng(cut_source)
                    ds_cut_sources[i, :] = ds_cut_source[:]
                 
                if not self._silent_exist(ds_cut_sources):
                    npz_num = npz_num + 1
                    mixture = np.sum(ds_cut_sources, axis=0)
                    file_path = self.save_folder_path + '{0}/{1}{2}'.format(mode, mode, npz_idx) 
                    self._to_npz(file_path, mixture, ds_cut_sources)
                    print("create npz:", self.save_folder_path + '{0}/{1}{2}'.format(mode, mode, npz_idx))
                    npz_idx = npz_idx + 1
                else:
                    print("silent file exists!")
                    
                end = time.time()
                print("excute_time:", end - start)
                    
                                  
    def make_test_tracks(self):
        subfolders = glob.glob(self.slakh_folder_path + 'test/*')
        npz_idx = 0
        black_list = []
        for track_path in subfolders:
            sources = glob.glob(track_path + '/stems/*')
            sources_list = []
            sources_shape = []
            sample_sources = random.sample(sources, self.inst_num)
            for source_path in sample_sources:
                data = self._read_as_mono(source_path)
                sources_list.append(self._downsampilng(data))
                sources_shape.append(data.shape)
                
            if not self._is_unique_shape(sources_shape):
                black_list.append(track_path)
                print('skip:{0}'.format(track_path))
                continue
            
            np_sources = np.array(sources_list)
            mixture = np.sum(np_sources, axis=0)
            file_path = self.save_folder_path + 'test/test{0}'.format(npz_idx) 
            self._to_npz(file_path, mixture, sample_sources)
            print("save file: {0}".format(file_path))
            npz_idx = npz_idx + 1
            
        print('black list:', black_list)
            
if __name__ == '__main__':
    random.seed(0)
    obj = SlakhToNpz(inst_num=2, train_npz_num=20000, valid_npz_num=2000)
    obj.make_train_tracks(mode='train')
    obj.make_train_tracks(mode='validation')
    obj.make_test_tracks()
