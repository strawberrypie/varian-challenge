import cv2
from skimage import measure, draw
import numpy as np

LA_THRESHOLD  = 9
THRESHOLD = 210

def perform_cv(test):
    test = cv2.convertScaleAbs(test, alpha=(255.0 / 65535.0))
    print(test.dtype)
    clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(4, 4))
    test = clahe.apply(test)
    test = cv2.convertScaleAbs(test, alpha=(255.0/65535.0))
#     print(np.max(test), np.min(test))
    test = cv2.morphologyEx(test, cv2.MORPH_OPEN, (7, 7), iterations=20)
    test = change_cont(test)
#     test = cv2.equalizeHist(test)
    test = cv2.fastNlMeansDenoisingColored(np.dstack((test, test, test)),None,10,10,7,21)[:,:,0]
    test = cv2.GaussianBlur(test, (7, 7), 1)
    test = cv2.dilate(test, (5, 5), iterations=10)
#     test = cv2.adaptiveThreshold(test, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#             cv2.THRESH_BINARY, 3, 2)
#     test = cv2.adaptiveThreshold(test,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,41,3)

    for contour in measure.find_contours(test > THRESHOLD, .5):
        ys, xs = zip(*contour)
        length = draw.polygon_perimeter(xs, ys)[0].shape[0]
        filled_polygon = draw.polygon(xs, ys)
        area = filled_polygon[0].shape[0]
        # plt.imshow(performed_cv > THRESHOLD)
        #     plt.plot(xs, ys)

        if length / np.sqrt(area) > LA_THRESHOLD:
            test[filled_polygon[1], filled_polygon[0]] = 0

    test[:, 240:270] = 0

    return test

def change_cont(test):
    b = 67
    c = 100
    test = cv2.addWeighted(test, 1. + c/127., test, 0, b-c)
    return test