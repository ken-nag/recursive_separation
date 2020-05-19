import torch
import torchaudio

class STFTModule():
    def __init__(self, stft_params):
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(hann(win_length)).to(torch.float32).to(device)
        
    def stft(self, x):
        return torch.stft(x, 
                          self.win_length, 
                          self.hop_length, 
                          self.win_length,
                          window=self.window)
    
    def istft(self, x):
        return torchaudio.functional.istft(x, 
                                           self.win_length, 
                                           self.hop_length,
                                           self.win_length, 
                                           window=self.window)
    