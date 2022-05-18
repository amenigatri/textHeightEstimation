from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torchvision import transforms


def height_from_connected_components(
    img_path: str, trans: transforms = transforms.CenterCrop((224, 224))
) -> (float, float):
    """
    Estimate the average height and median value of the text size from the input image.
    Part of the code is written following the example in
    "https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/"

    :param img_path: the path of the input image.
    :param trans: a transformation used to crop the input image at the center.
    :return: a tuple containing respectively the average height and median value of the text size from the input image.
    """
    image = Image.open(img_path)
    # image.show()
    p_img = trans(image)
    # convert pil image to cv2 image
    c_img = np.array(p_img)
    c_img = c_img[:, ::-1].copy()
    # grayscaling
    gray_im = c_img.convert('1')
    thresh = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # split the funct
    # apply connected component analysis to the thresholded image
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # loop over the number of unique connected component labels, skipping over the first label
    # (as label zero is the background)
    heights = []
    for i in range(1, num_labels):
        # extract the height for the current label
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h >= 5:
            heights.append(stats[i, cv2.CC_STAT_HEIGHT])
    # calculate the average height
    try:
        avg_height = sum(heights) / len(heights)
    except ZeroDivisionError:
        avg_height = 0
    # get the median height
    mid_height = np.median(heights)
    return avg_height, mid_height


def heights_from_all_images(
    files_list: pd.Series,
    dir_path: str,
    trans: transforms = transforms.CenterCrop((224, 224)),
) -> (list, list):
    """
    Estimate the average height and the median value of the text size for each input image from the files list.

    :param trans: a transformation used to crop the input image at the center.
    :param files_list: a pandas series contaning the names of the document images.
    :param dir_path: the path of the directory containing the document images.
    :return: a tuple of lists containing respectively the average heights and median values which correspond to the
    input images.
    """
    avg_heights = []
    medians = []
    for _, doc in enumerate(files_list):
        # Get the image path
        img_path = dir_path / Path(doc)
        # Calcualte the average height with connected components
        avg_h, mid_h = height_from_connected_components(img_path=img_path, trans=trans)
        avg_heights.append(avg_h)
        medians.append(mid_h)

    return avg_heights, medians
