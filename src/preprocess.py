import argparse
import dicom
import numpy as np
import os
from skimage import draw
import sys


def slice_loc_safe(img):
    try:
        return -img.SliceLocation
    except AttributeError:
        return 1000000


def process_test(directory):
    filenames = os.listdir(directory)

    uid_filename_dict = dict()
    for filename in filenames:
        dicom_image = dicom.read_file(os.path.join(directory, filename))
        try:
            uid_filename_dict[dicom_image.SOPInstanceUID] = filename
        except AttributeError:
            print("No UID in file {}".format(filename))
            sys.exit(1)

    mr_image_filenames = [v for v in uid_filename_dict.values() if "MR." in v]
    mr_images = [dicom.read_file(os.path.join(directory, f)) for f in mr_image_filenames]
    mr_images = sorted(mr_images, key=slice_loc_safe)

    if len(set([img.pixel_array.shape for img in mr_images])) != 1:
        print("Images of different sizes, exiting!")
        sys.exit(1)

    cur_x = np.zeros((len(mr_images), mr_images[0].pixel_array.shape[0], mr_images[0].pixel_array.shape[1]))

    for IDX in range(len(mr_images)):
        dicom_image = mr_images[IDX]
        cur_x[IDX] = dicom_image.pixel_array

    return cur_x


def process_train(directory, plan_filename):
    plan = dicom.read_file(plan_filename)

    try:
        uid_contour_dict = dict()
        for roi in plan.ROIContourSequence:
            for contour in roi.get("ContourSequence", []):
                assert len(contour.ContourImageSequence) == 1
                cur_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
                uid_contour_dict[cur_uid] = contour.ContourData
    except AttributeError:
        print("No labelling data in the PLAN file!")
        sys.exit(1)

    filenames = os.listdir(directory)

    uid_filename_dict = dict()
    for filename in filenames:
        dicom_image = dicom.read_file(os.path.join(directory, filename))
        try:
            uid_filename_dict[dicom_image.SOPInstanceUID] = filename
        except AttributeError:
            print("No UID in file {}".format(filename))
            sys.exit(1)

    mr_image_filenames = [v for v in uid_filename_dict.values() if "MR." in v]
    mr_images = [dicom.read_file(os.path.join(directory, f)) for f in mr_image_filenames]
    mr_images = sorted(mr_images, key=slice_loc_safe)

    if len(set([img.pixel_array.shape for img in mr_images])) != 1:
        print("Images of different sizes, exiting!")
        sys.exit(1)

    cur_x = np.zeros((len(mr_images), mr_images[0].pixel_array.shape[0], mr_images[0].pixel_array.shape[1]))
    cur_y = np.zeros((len(mr_images), mr_images[0].pixel_array.shape[0], mr_images[0].pixel_array.shape[1]))

    for IDX in range(len(mr_images)):
        dicom_image = mr_images[IDX]
        cur_x[IDX] = dicom_image.pixel_array
        mask = np.zeros_like(dicom_image.pixel_array)
        scale_x = scale_y = 1
        shift_x = shift_y = shift_z = 0

        if dicom_image.get("PixelSpacing") is None:
            print("No Pixel spacing, image #{}".format(IDX))

        if dicom_image.get("ImagePositionPatient") is None:
            print("No position, image #{}".format(IDX))

        scale_x, scale_y = dicom_image.get("PixelSpacing", [scale_x, scale_y])
        shift_x, shift_y, shift_z = dicom_image.get("ImagePositionPatient", [shift_x, shift_y, shift_z])

        cur_uid = mr_images[IDX].get("SOPInstanceUID")
        if cur_uid in uid_contour_dict:
            contour_data = uid_contour_dict[cur_uid]

            xs = np.array(contour_data[::3]) - shift_x
            ys = np.array(contour_data[1::3]) - shift_y

            xs /= scale_x
            ys /= scale_y

            poly_idx = draw.polygon(ys, xs, shape=mask.shape)
            mask[poly_idx[0], poly_idx[1]] = 1

        cur_y[IDX] = mask
    return cur_x, cur_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True, help="path to dir with dicom files")
    parser.add_argument("-p", "--plan", type=str, required=True, help="filename of file with labelling data")
    parser.add_argument("-o", "--out", type=str, required=True, help="path to .npz file with two np.arrays inside: X and Y")

    args = parser.parse_args()

    cur_x, cur_y = process_train(args.dir, args.plan)

    np.savez_compressed(args.out, X=cur_x, Y=cur_y)
