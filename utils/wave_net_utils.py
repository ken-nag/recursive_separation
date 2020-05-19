import torch

class WaveUNetUtils():
    def centre_crop(self, us_feature, ds_feature):
        d = ds_feature.shape[2] - us_feature.shape[2]
        return ds_feature[:,:,d//2:-d+(d//2)]
    
    def cat_operater(self, us_feature, ds_feature):
        cropped_feature = self.centre_crop(us_feature, ds_feature)
        return torch.cat([cropped_feature, us_feature], dim=1)
    