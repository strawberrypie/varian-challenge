
class SegAugmentWrapper(object):
    """
    Wraps augmentations for both image and sementations mask to Catalyst format.
    """
    
    def __init__(self, augment_fn, image_key='features', mask_key='targets'):
        """augment_fn: (image, mask) -> (aug_image, aug_mask)"""
        self.augment_fn = augment_fn
        self.image_key = image_key
        self.mask_key = mask_key
        
    def __call__(self, dict_):
        dict_[self.image_key], dict_[self.mask_key] = self.augment_fn(
            dict_[self.image_key], dict_[self.mask_key])
        return dict_

class SegAlbumWrapper(object):
    """
    Wraps augemtations for both image and segmentation mask to Catalyst format
    from albumentations library.
    """
    def __init__(self, augment_fn, image_key='features', mask_key='targets'):
        """augment_fn: (image, mask) -> (aug_image, aug_mask)"""
        self.augment_fn = augment_fn
        self.image_key = image_key
        self.mask_key = mask_key
        
    def __call__(self, dict_):
        augmented = self.augment_fn(
            image=dict_[self.image_key],
            mask=dict_[self.mask_key]
        )
        dict_[self.image_key], dict_[self.mask_key] = augmented['image'], augmented['mask']
        return dict_