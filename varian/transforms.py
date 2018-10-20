import cv2
import numpy as np

import torch
from torchvision.transforms import functional as F

import math
import random


class ComposeJoint():
    """
    Joiner of transformators
    """
    def __init__(self, transforms):
        """
        Joins list of transformators and applies them iteratively
        :param transforms: array-like of callables
        """
        self._transforms = transforms
    
    def __call__(self, sample):
        """
        Iterates over all transformators and applies them sequentially
        :param sample: dictionary with image, labels and name properties
        :return: processed dictionary with image, labels and name properties
        """
        img = sample['image']
        labels = sample['labels']
        img_name = sample['name']
        
        for t in self._transforms:
            img, labels = t(img, labels)
        
        return {'image': img, 'labels': labels, 'name': img_name}
    
    
class Pad():
    """
    Tranformator for image padding
    """
    def __call__(self, img, mask=None):
        """
        Pads an image and a mask to form square shape (borders are replicated)
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        img = Pad.pad_image(img, cv2.BORDER_REPLICATE)
        
        if mask is not None:
            mask = Pad.pad_image(mask, cv2.BORDER_REPLICATE)
        
        return img, mask
    
    @staticmethod
    def pad_image(img, border_type, value=None):
        """
        Pads an image to form square shape with given border type
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param border_type: OpenCV constant meaning border type
            (starts with cv2.BORDER_*)
        :param value: (Optional) fill value in case border type is constant
        :return: padded image
        """
        print(img.shape)
        h, w = img.shape[:2]

        if w > h:
            padding = int((((w - h) / h) / 2) * h)
            dst = cv2.copyMakeBorder(img, padding, padding, 0, 0, border_type, value=value)
        elif w < h:
            padding = int((((h - w) / w) / 2) * w)
            dst = cv2.copyMakeBorder(img, 0, 0, padding, padding, border_type, value=value)
        else:
            dst = img

        return dst


class GreedyPad():
    def __init__(self, binary_thresh=100, offset=20, prob=0.5):
        """
        Transformator for padding image and mask to actual size of the molecule
            (bounding box of the molecule is estimated by some heuristics)
        :param binary_thresh: threshold value used to binarize initial image
        :param offset: a coefficient used for calculation of margins
            between molecule and image borders
        :param prob: a probability of the operation to be applied
        """
        self.binary_thresh = binary_thresh
        self.offset = offset
        self.prob = prob
    
    def __call__(self, img, mask=None):
        """
        Callable for the padding operation
        :param img: numpy.ndarray of shape (height, width, 3) representing image
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        if random.random() < self.prob:
            h, w = img.shape[:2]
            
            # In case of bad lighting conditions the image and mask remain the same
            top, bottom, left, right = 0, h, 0, w

            offset = min(w, h) // self.offset
            m = np.all(img < self.binary_thresh, axis=2).astype(np.uint8) * 255
            
            # Configure the most appropriate kernel size for noise reduction (for dataset images it's equal to 7)
            a = max(w, h)
            ksize = a // 600 + 1 if (a // 600) % 2 == 0 else a // 600
            
            # Apply erode operation in order to get rid of noise
            kernel = np.ones((ksize, ksize), dtype=np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_ERODE, kernel)
            
            # Calculate margins to remove outside regions from image
            idxs = np.where(m == 255)
            if len(idxs[0]) > 0 and len(idxs[1]) > 0:
                top, bottom, left, right = np.min(idxs[0]), np.max(idxs[0]), np.min(idxs[1]), np.max(idxs[1])

            # Remove outside regions from image and turn image to square shape
            img = img[max(top - offset, 0):min(bottom + offset, h), max(left - offset, 0):min(right + offset, w)]
            img = Pad.pad_image(img, cv2.BORDER_REPLICATE)

            # Do the same for mask
            if mask is not None:
                mask = mask[max(top - offset, 0):min(bottom + offset, h), max(left - offset, 0):min(right + offset, w)]
                mask = Pad.pad_image(mask, cv2.BORDER_REPLICATE)
        
        return img, mask


class Resize():
    def __init__(self, size):
        """
        Transformator for resizing image and mask
        :param size: tuple (int, int) representing size of the output image
        """
        self._h = size[0]
        self._w = size[1]
    
    def __call__(self, img, mask=None):
        """
        Callable for resizing operation
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        img = cv2.resize(img, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        
        if mask is not None:
            mask = cv2.resize(mask, (self._w, self._h), interpolation=cv2.INTER_NEAREST)
        
        return img, mask


class Rescale():
    def __init__(self, factor):
        """
        Transformator for image scaling
        :param factor: value by which the image will be multiplied
        """
        self._factor = factor
    
    def __call__(self, img, mask=None):
        """
        Callabe for the scaling operation
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        img = img.astype(np.float32) * self._factor
        
        return img, mask


class NumpyToTensor():
    """
    Transformator for converting numpy.ndarray object representing image
        and mask into image and mask of type torch.Tensor
        with channel dimension shifted to the first dimension
    """
    def __call__(self, img, mask=None):
        """
        Callable for the convertion operation
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        if len(img.shape) >= 3:
            img = img.transpose(2, 0, 1)
        else:
            img = np.expand_dims(img, 0)
        img = torch.Tensor(img.copy())
        
        if mask is not None:
            mask = torch.Tensor(mask.copy())
        
        return img, mask


class Normalize():
    def __init__(self, mean, std):
        """
        Transformator for standard image rescaling
        """
        self._mean = mean
        self._std = std
    
    def __call__(self, img, mask=None):
        """
        Callable for the standard rescaling
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        img = F.normalize(img, self._mean, self._std)
        
        return img, mask


class PdfImagesCropper():
    def __init__(self, c=0.03):
        """
        Transformator for cropping black borders of images extracted from PDF
        :param c: percent of image size to be cropped out
            from the image for each side
        """
        self._c = c
    
    def __call__(self, img, mask=None):
        """
        Callable for cropping operation
        :param img: numpy.ndarray of shape (height, width, C) representing image
            where C - number of channels
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        c = self._c
        h, w = img.shape[:2]
        img = img[int(c * h):h - int(c * h), int(c * w):w - int(c * w)]
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_NEAREST)
        
        return img, mask
    

# The next set of tranformators are for binary images only
    
class InvertBinaryImage():
    """
    Transformator for inverting binary values of an image
    """
    def __call__(self, img, mask=None):
        """
        Callable of invertion operation
        :param img: numpy.ndarray of shape (height, width) representing image
            (image must be of numpy.uint8 type and have only 0 and 255 values)
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        img = 255 - img
        
        return img, mask


class RandomDilation():
    def __init__(self, ksize_range, prob=0.5):
        """
        Transformator for applying dilation morphology operation
            with random kernel size from the given range
        :param ksize_range: tuple (int, int) representing range
            from which kernel size for the operation will be drawn
        :param prob: probability of operation to be applied
        """
        self._prob = prob
        self._ksize_range = ksize_range
    
    def __call__(self, img, mask=None):
        """
        Callable for random dilation operation
        :param img: numpy.ndarray of shape (height, width) representing image
            (image must be of numpy.uint8 type and have only 0 and 255 values)
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        if self._prob > np.random.rand():
            ksize = np.random.randint(self._ksize_range[0], self._ksize_range[1] + 1)
            img = self._apply_dilation(img, ksize)

            if mask is not None:
                mask = self._apply_dilation(mask, ksize)
        
        return img, mask
    
    def _apply_dilation(self, img, ksize):
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        cv2.circle(kernel, (ksize // 2, ksize // 2), ksize // 2, color=255, thickness=-1)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        
        return img


class RandomMorphologyKernel():
    def __init__(self, prob=0.5):
        """
        Transformator for applying morphology operation with random kernel
        :param prob: probability of operation to be applied
        """
        self._prob = prob
    
    def __call__(self, img, mask=None):
        """
        Callable for the morphology operation -
            kernel size is estimated based on image size, the kernel has Bernulli distribution
            with p=0.5, operation is morphology closing
        :param img: numpy.ndarray of shape (height, width) representing image
            (image must be of numpy.uint8 type and have only 0 and 255 values)
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        if self._prob > np.random.rand():
            ksize = max(img.shape[:2]) // 100
            kernel = np.random.binomial(1, 0.5, (ksize, ksize)).astype(np.uint8) * 255
            kernel[ksize // 2 - ksize // 4: ksize // 2 + ksize // 4 + 1,
                   ksize // 2 - ksize // 4: ksize // 2 + ksize // 4 + 1] = 255
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return img, mask


class RandomNoise():
    def __init__(self, prob_range=(0.01, 0.03), prob=0.5):
        """
        Transformator for random noise generation (white spots appear with some probability)
        :param prob_range: tuple (float, float) representing range of Bernulli probability
            with which the white spots appear in the image
        :param prob: probability of operation to be applied
        """
        self._prob = prob
        self._prob_range = prob_range
    
    def __call__(self, img, mask=None):
        """
        Callable for random noise operation
        :param img: numpy.ndarray of shape (height, width) representing image
            (image must be of numpy.uint8 type and have only 0 and 255 values)
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        if self._prob > np.random.rand():
            prob_range = self._prob_range
            p = np.random.rand() * (prob_range[1] - prob_range[0]) + prob_range[0]
            m = np.random.binomial(1, p, img.shape[:2]).astype(np.uint8) * 255
            img[m > 0] = 255
        
        return img, mask


class RestoreBinary():
    """
    Transformator that is useful after applying affine operations on binary image
        (after those operations binary values aren't pure so we have to binarize image again)
    """
    def __call__(self, img, mask=None):
        """
        Callable for binary restore operation
        :param img: numpy.ndarray of shape (height, width) representing image
            (image must be of numpy.uint8 type and have only 0 and 255 values)
        :param mask: (Optional) numpy.ndarray of shape (height, width) representing labels
        :return: img, mask - processed results
        """
        img[img < 255] = 0
        
        return img, mask


# The remaining is from https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/transforms.py

class OneOf:
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class VerticalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Transpose:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0)
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):        
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)            
            if mask is not None:
                mask = np.rot90(mask, factor)            
        return img.copy(), mask.copy() # throws error without .copy()


class Rotate:
    def __init__(self, limit=90, prob=.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (width, height),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REPLICATE)

        return img, mask


class Shift:
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width = img.shape[:2]
            y1 = limit+1+dy
            y2 = y1 + height
            x1 = limit+1+dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit+1, limit+1, limit+1, limit+1, borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2]
            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit+1, limit+1, limit+1, limit+1, borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2]

        return img, mask


class ShiftScale:
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        limit = self.limit
        if random.random() < self.prob:
            height, width = img.shape[:2]
            assert(width == height)
            size0 = width
            size1 = width + 2*limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1-size))
            dy = round(random.uniform(0, size1-size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            fill_value = tuple(np.max(img, axis=(0, 1)))
            
            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, cv2.BORDER_CONSTANT, value=fill_value)
            img = (img1[y1:y2, x1:x2] if size == size0
                   else cv2.resize(img1[y1:y2, x1:x2], (size0, size0), interpolation=cv2.INTER_LINEAR))

            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit, cv2.BORDER_CONSTANT, value=(0,))
                mask = (msk1[y1:y2, x1:x2] if size == size0
                        else cv2.resize(msk1[y1:y2, x1:x2], (size0, size0), interpolation=cv2.INTER_NEAREST))

        return img, mask


class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width = img.shape[:2]

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1-self.scale_limit, 1+self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle/180*math.pi) * scale
            ss = math.sin(angle/180*math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
            box1 = box0 - np.array([width/2, height/2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx, height/2+dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(mask, mat, (width, height),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class CenterCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2, :]
        if mask is not None:
            mask = mask[y1:y2, x1:x2, ...]

        return img, mask


class Distort1:
    """"
    ## unconverntional augmnet ################################################################################3
    ## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

    ## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
    ## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
    ## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

    ## barrel\pincushion distortion
    """
    def __init__(self, distort_limit=0.35, shift_limit=0.25, prob=0.5):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width = img.shape[:2]

            if 0:
                img = img.copy()
                for x in range(0, width, 10):
                    cv2.line(img, (x, 0), (x, height), (1, 1, 1), 1)
                for y in range(0, height, 10):
                    cv2.line(img, (0, y), (width, y), (1, 1, 1), 1)

            k = random.uniform(-self.distort_limit, self.distort_limit) * 0.00001
            dx = random.uniform(-self.shift_limit, self.shift_limit) * width
            dy = random.uniform(-self.shift_limit, self.shift_limit) * height

            #  map_x, map_y =
            # cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
            # https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
            # https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
            x, y = np.mgrid[0:width:1, 0:height:1]
            x = x.astype(np.float32) - width/2 - dx
            y = y.astype(np.float32) - height/2 - dy
            theta = np.arctan2(y, x)
            d = (x*x + y*y)**0.5
            r = d*(1+k*d*d)
            map_x = r*np.cos(theta) + width/2 + dx
            map_y = r*np.sin(theta) + height/2 + dy

            img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        return img, mask


class Distort2:
    """
    #http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    ## grid distortion
    """
    def __init__(self, num_steps=10, distort_limit=0.2, prob=0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width = img.shape[:2]

            x_step = width // self.num_steps
            xx = np.zeros(width, np.float32)
            prev = 0
            for x in range(0, width, x_step):
                start = x
                end = x + x_step
                if end > width:
                    end = width
                    cur = width
                else:
                    cur = prev + x_step*(1+random.uniform(-self.distort_limit, self.distort_limit))

                xx[start:end] = np.linspace(prev, cur, end-start)
                prev = cur

            y_step = height // self.num_steps
            yy = np.zeros(height, np.float32)
            prev = 0
            for y in range(0, height, y_step):
                start = y
                end = y + y_step
                if end > width:
                    end = height
                    cur = height
                else:
                    cur = prev + y_step*(1+random.uniform(-self.distort_limit, self.distort_limit))

                yy[start:end] = np.linspace(prev, cur, end-start)
                prev = cur

            map_x, map_y = np.meshgrid(xx, yy)
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)
            img = cv2.remap(img, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.remap(mask, map_x, map_y,
                                 interpolation=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class RandomFilter:
    """
    blur sharpen, etc
    """
    def __init__(self, limit=.5, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        img = img.copy()
        
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(0, 1)
            kernel = np.ones((3, 3), np.float32)/9 * 0.2

            colored = img[..., :3]
            colored = alpha * cv2.filter2D(colored, -1, kernel) + (1-alpha) * colored
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(colored, dtype, maxval)

        return img, mask


# https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
# color augmentation

# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        img = img.copy()
        
        if random.random() < self.prob:
            alpha = 1.0 + self.limit*random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[...,:3], dtype, maxval)
        return img, mask


class RandomContrast:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        img = img.copy()
        
        if random.random() < self.prob:
            alpha = 1.0 + self.limit*random.uniform(-1, 1)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
        return img, mask


class RandomSaturation:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        img = img.copy()
        
        # dont work :(
        if random.random() < self.prob:
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[..., :3] = alpha * img[..., :3] + (1.0 - alpha) * gray
            img[..., :3] = clip(img[..., :3], dtype, maxval)
        return img, mask

class RandomHueSaturationValue:
    def __init__(self, hue_shift_limit=(-10, 10), sat_shift_limit=(-25, 25), val_shift_limit=(-25, 25), prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image, mask=None):
        image = image.copy()
        
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, mask

class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im, mask=None):
        im = im.copy()
        
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output, mask


def augment(x, mask=None, prob=0.5):
    return DualCompose([
        OneOrOther(
            *(OneOf([
                Distort1(distort_limit=0.05, shift_limit=0.05),
                Distort2(num_steps=2, distort_limit=0.05)]),
              ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=45)), prob=prob),
        RandomFlip(prob=0.5),
        Transpose(prob=0.5),
        ImageOnly(RandomContrast(limit=0.2, prob=0.5)),
        ImageOnly(RandomFilter(limit=0.5, prob=0.2)),
    ])(x, mask)


def augment_a_little(x, mask=None, prob=.5):
    return DualCompose([
        HorizontalFlip(prob=.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=5, prob=.75)
    ])(x, mask)


def augment_color(x, mask=None, prob=.5):
    return DualCompose([
        HorizontalFlip(prob=.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=5, prob=.75),
        ImageOnly(RandomBrightness()),
        ImageOnly(RandomHueSaturationValue())
    ])(x, mask)
