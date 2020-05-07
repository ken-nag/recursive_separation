import torch

class OrPit():
    def _cal_err(self, est_source, est_res_source, true_source, true_res_source, N):
        return self._si_snr(est_source, true_source) + (1/(N-1)) * self._si_snr(est_res_source, true_res_source)
    
    def _si_snr(self, est_source, true_source):
        normalize_est = self._mean_normalize(est_source)
        normalize_true = self._mean_normalize(true_source)
        target = (torch.dot(normalize_est, normalize_true) / torch.dot(normalize_true, normalize_true)) * normalize_true
        noise = normalize_est - target
        return 10 * torch.log10(torch.dot(target, target) / torch.dot(noise, noise))
    
   
    def _mean_normalize(self, x):
        means = torch.mean(x, axis=1)
        return torch.div(x, means.view(means.shape(-1,1)))
    
    def __call__(self, est_sources, est_res_sources, true_sources, true_res_sources, N):
        # todo:batchで計算
        err_list = []
        
        for est_source, est_res_source in zip(est_sources, est_res_sources):
            tmp_err_list = []
            
            for true_source, true_res_source in zip(true_sources, true_res_sources):
                err = self._cal_err(est_source, est_res_source, true_source, true_res_sources, N)
                tmp_err_list.append(err)
            
            err_list.append(min(tmp_err_list))
            
        return torch.mean(err_list)
