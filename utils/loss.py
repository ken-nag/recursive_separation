import torch

class OrPit():
    @classmethod
    def loss(cls, est_x, true_x, est_res, true_res, N):
        return _si_snr(est_x, true_x) + (1/(N-1)) * _si_snr(est_res, true_res)
        
    def _si_snr(self, est_x, true_x):
        norm_est = _mean_normalize(est_x)
        norm_true = _mean_normalize(true_x)
        target = 
        noise = 
        return 10
    
    def _mean_normalize(self, x):
        pass
    
    def 