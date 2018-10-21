import cv2
import numpy as np

def mk_mask(test):
    processed = perform_cv2(test.astype(np.uint16))
    return filter_components(processed)

def perform_cv2(test):
    cur_img_norm = test/test.max()
    thresholded = cur_img_norm > np.mean(cur_img_norm)*np.pi
    thresholded = cv2.erode(thresholded.astype(np.uint16), (30, 30), iterations=10)
    thresholded = cv2.GaussianBlur(thresholded, (31, 31), 1)
    thresholded = cv2.morphologyEx(thresholded.astype(np.uint8), cv2.MORPH_OPEN, (30, 30), iterations=10)
    thresholded = cv2.GaussianBlur(thresholded, (11, 11), 10)
    return thresholded

def filter_components(test):
    ret, thresh = cv2.threshold(test,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    num_labels = output[0]
    label_map = output[1]
    label_stats = output[2]
    label_width = label_stats[:, cv2.CC_STAT_WIDTH]
    label_height = label_stats[:, cv2.CC_STAT_HEIGHT]
    label_area = label_stats[:, cv2.CC_STAT_AREA]
    ratio = label_width / label_height

    bad_inds = []
    for index in range(num_labels):
        label_area_percentage = label_area[index] / (label_map.shape[0] * label_map.shape[1])
        if label_area_percentage >= 0.01 or label_area_percentage <= 0.0002:
            bad_inds.append(index)
            continue
        if ratio[index] >= 1.3 or ratio[index] <= 0.7:
            bad_inds.append(index)
            continue

    for bad_ind in bad_inds:
        label_map[label_map == bad_ind] = 0

    label_map = cv2.dilate(label_map.astype(np.uint8), (3, 4), iterations=10)
    label_map[100:150, 250:275] = 0
    label_map[400:470, 250:275] = 0
    print(type(label_map))
    return label_map

#
# processed = perform_cv2(image.astype(np.uint16))
# final_result = filter_components(processed)
