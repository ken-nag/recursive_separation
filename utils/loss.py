import torch

class SiSNRLoss():
    def __init__(self):
        self.epsilon = 1e-10
        
    def _cal_err(self, est_sources, est_res_sources, true_sources, true_res_sources, N):
        return self._si_snr(est_sources, true_sources) + (1/(N-1)) * self._si_snr(est_res_sources, true_res_sources)
    
    def _si_snr(self, est_sources, true_sources):
        # 定義を見直す
        normalize_est = self._mean_normalize(est_sources)
        normalize_true = self._mean_normalize(true_sources)
        target = self._inner_product(normalize_est, normalize_true) / (self._inner_product(normalize_true, normalize_true) + self.epsilon) * normalize_true
        noise = target - normalize_est
        return -10 * torch.log10(self._inner_product(target, target) / (self._inner_product(noise, noise) + self.epsilon))
    
    def _inner_product(self, a, b):
        return torch.sum(a*b, dim=2, keepdim=True, dtype=torch.float)
   
    def _mean_normalize(self, x):
        means = torch.mean(torch.abs(x), axis=2, keepdim=True)
        return x / (means + self.epsilon)
    
    def __call__(self, est_sources, est_res_sources, true_sources, true_res_sources, inst_num):
        batch_size = est_sources.shape[0]
        est_sources = est_sources.repeat(1, inst_num, 1)
        est_res_sources = est_res_sources.repeat(1, inst_num, 1)
        err = self._cal_err(est_sources, est_res_sources, true_sources, true_res_sources, inst_num)
        min_loss, _ = torch.min(err.squeeze(2), dim=1)
        loss  = torch.sum(min_loss, dim=0) / batch_size
        return loss 
    