import numpy as np
import matplotlib.pyplot as plt

    
def show_TF_domein_result(loss, mixture, mask, estimate, vmin=-60, eps=1e-10):
    mixture = mixture.cpu().clone().numpy()
    mask = mask.cpu().clone().numpy()
    estimate = estimate.cpu().clone().numpy()
    
    vmax = 20*np.log10(np.max(mixture))-10
    vmin += vmax
    
    mask_max = 20*np.log10(np.max(mask))-10
    mask_min = mask_max - 60
    
    fig, axes = plt.subplots(2,2, figsize=(12,8))

    axes[0,0].plot(loss)
    axes[0,0].set_title('loss')
    axes[0,0].axis('auto')
    
    axes[0,1].imshow(20*np.log10(np.flipud(mixture)+eps), vmax=vmax, vmin=vmin, aspect="auto")
    axes[0,1].set_title('mixture')
    
    axes[1,0].imshow(20*np.log10(np.flipud(mask)+eps), vmax=mask_max, vmin=mask_min, aspect="auto")
    axes[1,0].set_title('mask')
    
    axes[1,1].imshow(20*np.log10(np.flipud(estimate)+eps), vmax=vmax, vmin=vmin, aspect="auto") 
    axes[1,1].set_title('estimate')

        
    plt.tight_layout()
        
    plt.show()
    plt.close()
