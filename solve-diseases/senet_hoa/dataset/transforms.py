import albumentations as A
import cv2
import numpy as np

class FilterSmallComponents(A.DualTransform):
    def __init__(self, min_size=4, value=1):
        super(FilterSmallComponents, self).__init__(always_apply=True, p=1)
        self.min_size = min_size
        self.value=value

    def apply_to_mask(self, mask, *args,**params):
        return self.filter_small_components(mask)
    def apply(self, image, *args,**params):
        return image
    def filter_small_components(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        new_mask = np.zeros_like(mask)

        for i in range(1, num_labels):  # Skip the background
            size = stats[i, cv2.CC_STAT_AREA]
            if size >= self.min_size:
                new_mask[labels == i] = self.value

        return new_mask
class MaskBinarize(A.DualTransform):
    def __init__(self, cutoff=0,value_max=1,value_min=0):
        super(MaskBinarize, self).__init__(always_apply=True, p=1)
        self.cutoff = cutoff
        self.value_max = value_max
        self.value_min = value_min
    def apply_to_mask(self, mask, *args,**params):
        return np.where(mask>self.cutoff,self.value_max,self.value_min)    
    
    def apply(self, image, *args,**params):
        return image

class Normalize(A.DualTransform):
    def __init__(self, max_val=255,min_val=0):
        super(Normalize, self).__init__(always_apply=True, p=1)
        self.max_val=max_val
        self.min_val=min_val

    def apply_to_mask(self, mask, *args,**params):
        return mask
    def apply(self, image, *args,**params):
        return (image-self.min_val)/(self.max_val-self.min_val)
