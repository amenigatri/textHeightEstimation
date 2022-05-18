from pathlib import Path
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
import hashlib
import torch as t
from torch.utils.data import DataLoader
from PIL import UnidentifiedImageError
from PIL import Image
from tqdm import tqdm
from random import random
import pickle
import collections
import matplotlib.pyplot as plt
from src.text_height_estimation.cnn_model import CNN
from src.text_height_estimation.connected_components import heights_from_all_images
from src.text_height_estimation.data_preprocessing import (
    load_document_images,
    TextDataset,
)


# size of the crops
crop_size = 320

# rough number of crops for each type group (might be approximate)
samples_per_class = 15000

# page images wider than this are resized to this
image_max_width = 1000

rnd_crop = transforms.RandomCrop(crop_size)


def count_samples_per_typegroup(training_csv_path: str):
    """
    Count the number of documents where each script group is used for the training.
    :param training_csv_path: path to a csv file of the training dataset where each document image is labeled with the
    presented script.
    :return: a dictionary containing each script and the number of documents using it for the training.
    """
    doc_script = get_documents_scripts_list(training_csv_path)
    print("Count samples per type group")
    # count samples per type group
    count = {}
    for sample in doc_script:
        if sample[1] in count:
            count[sample[1]] += 1
        else:
            count[sample[1]] = 1

    return count


def plot_scripts(count: dict):
    """
    Plot the distribution of documents by script group
    :param count: dictionary containing the number of documnets image for each script group
    :return:
    """
    # store the amount of documents for each script
    script_list = []
    n_docs = []
    # sort count dictionary by script type
    count_sorted = collections.OrderedDict(sorted(count.items()))
    for key, value in count_sorted.items():
        script_list.append(key)
        n_docs.append(value)

    # plot the distribution of scripts
    x = list(range(1, len(script_list) + 1))
    new_x = [3 * i for i in x]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 10})
    plt.bar(new_x, n_docs)
    plt.xticks(new_x, script_list)
    plt.title("Distribution of scripts")
    plt.tight_layout()
    plt.show()


def create_training_validation_directories(
    count: dict,
    data_dir_path: str,
):
    """
    Create two directories to store the training and validation datasets for the training process for the tree approches.
    :param count: a dictionary containing each script and the number of documents using it.
    :param data_dir_path: the path to the directory containing scriptgroupdataset labels and documents to be used for the
    script script_recognition task.
    :return:
    """
    print("Create training and validation directories")
    for resizing_mode in ["baseline", "binarization", "cnn", "resnet18"]:

        training_dir_path = data_dir_path / Path(
            f"./{resizing_mode}/typegroupdataset-training"
        )
        training_dir_path.mkdir(parents=True, exist_ok=True)

        validation_dir_path = data_dir_path / Path(
            f"./{resizing_mode}/typegroupdataset-validation"
        )
        validation_dir_path.mkdir(parents=True, exist_ok=True)

        for main_group_script in count:
            path = training_dir_path / Path(f"./{main_group_script}")
            path.mkdir(parents=True, exist_ok=True)

            path = validation_dir_path / Path(f"./{main_group_script}")
            path.mkdir(parents=True, exist_ok=True)


def get_documents_scripts_list(training_csv_path: str):
    """
    Get list of the documents in the training set and ther correspnding script labels.
    :param training_csv_path: path to a csv file of the training dataset where each document image is labeled with the
    presented script.
    :return: a list of lists of documents with teir corresponding script label.
    """
    doc_df = pd.read_csv(training_csv_path)
    doc_df.drop(columns=["DATE_TYPE"], inplace=True)
    doc_script = doc_df.values.tolist()
    return doc_script


def get_heights_for_cnn_resizing(documents: list, data_dir_path: str, cnn_path: str):
    """
    Estimate heights of the documents using the trained CNN model.
    :param documents: list of documents for the training and validation process.
    :param data_dir_path: the path to the directory containing scriptgroupdataset labels and documents to be used for the
    script script_recognition task.
    :param cnn_path:  path to the trained cnn model for text height estimation.
    :return: list of estimated heights corresponding to the input document images.
    """
    # load trained cnn
    print("Load trained CNN")
    model = CNN()
    model.load_state_dict(t.load(cnn_path))

    # create any array for the heights
    y_train = np.zeros(len(documents))
    y_train = t.Tensor(y_train)
    print("Load data")
    x_train = load_document_images(documents, data_dir_path)
    training_dataset = TextDataset(x_train, y_train)
    trainingLoader = DataLoader(training_dataset, batch_size=100, shuffle=False)
    # save predicted heights
    heights = []
    print("Get heights with CNN")
    with t.no_grad():
        for _, data in enumerate(trainingLoader):
            input_documents, _ = data
            # calculate outputs by running input_documents through the network
            predicted_heights = model(input_documents)
            # convertt tensor to list and append each value to heights list
            for _, h in enumerate(predicted_heights.flatten().tolist()):
                heights.append(h)

    return heights


def get_heights_for_resnet18_resizing(documents: list, data_dir_path: str, resnet18_path: str):
    """
    Estimate heights of the documents using the trained ResNet18 model.
    :param documents: list of documents for the training and validation process.
    :param data_dir_path: the path to the directory containing scriptgroupdataset labels and documents to be used for the
    script script_recognition task.
    :param resnet18_path:  path to the trained resnet18 model for text height estimation.
    :return: list of estimated heights corresponding to the input document images.
    """
    # load trained resnet18
    print("Load trained ResNet18")
    model = resnet18(num_classes=1)
    model.conv1 = t.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
    model.load_state_dict(t.load(resnet18_path))

    # create any array for the heights
    y_train = np.zeros(len(documents))
    y_train = t.Tensor(y_train)
    print("Load data")
    x_train = load_document_images(documents, data_dir_path)
    training_dataset = TextDataset(x_train, y_train)
    trainingLoader = DataLoader(training_dataset, batch_size=100, shuffle=False)
    # save predicted heights
    heights = []
    print("Get heights with ResNet18")
    with t.no_grad():
        for _, data in enumerate(trainingLoader):
            input_documents, _ = data
            # calculate outputs by running input_documents through the network
            predicted_heights = model(input_documents)
            # convertt tensor to list and append each value to heights list
            for _, h in enumerate(predicted_heights.flatten().tolist()):
                heights.append(h)

    return heights


def resizing_images(im, resizing_mode, norm_height, heights, indx, fname):
    """
    Resize the input image such as the text height will be equal to the normalization height.
    :param im: the input image to be resized.
    :param resizing_mode: the resizing mode used to estimate the normalization height. It can be either baseline, binarization or cnn.

    :param norm_height: the value of normalization height.
    :param heights: dictionary containing theestimated heights for binarization and cnn resizing
    modes.
    :param indx: the index used to get the estimated height corresponding to the nput image
    :param fname: the name under which the input image is saved in the data directory.
    :return: the resized image and a boolean to indicate if an unidentified image error occured while resizing the
    image.
    """
    error_bool = False
    new_im = im
    if resizing_mode in ["binarization", "cnn", "resnet18"]:
        # resize image height such as the text height will be the norm_height
        try:
            new_im = im.resize(
                (
                    round(
                        (im.size[0] * norm_height)
                        / heights[resizing_mode][indx]
                    ),
                    round(
                        (im.size[1] * norm_height)
                        / heights[resizing_mode][indx]
                    ),
                ),
                Image.BILINEAR,
            )
        except UnidentifiedImageError:
            print(f"Could not resize {fname}")
            error_bool = True
    # for the baseline
    if resizing_mode == "baseline":
        if im.size[0] > image_max_width:
            try:
                new_im = im.resize(
                    (
                        image_max_width,
                        round((im.size[1] * image_max_width) / float(im.size[0])),
                    ),
                    Image.BILINEAR,
                )
            except UnidentifiedImageError:
                print(f"Could not resize {fname}")
                error_bool = True
    return new_im, error_bool


def split_training_validation_sets(
    training_csv_path: str,
    data_dir_path: str,
    cnn_path: str,
    resnet18_path: str,
    norm_height: int,
):
    """
    Create the training and validation sets for the baseline, binarization- and CNN-based algorithms.
    :param training_csv_path: path to a csv file of the training dataset where each document image is labeled with the
    presented script.
    :param data_dir_path: the path to the directory containing scriptgroupdataset labels and documents to be used for the
    script script_recognition task.
    :param cnn_path: path to the trained cnn model for text height estimation.
    :param resnet18_path: path to the trained cnn model for text height estimation.
    :param norm_height: value of normalization height
    :return:
    """
    doc_script = get_documents_scripts_list(training_csv_path)
    documents = []
    for sample in doc_script:
        documents.append(sample[0])
    # get heights with cnn
    heights_cnn = get_heights_for_cnn_resizing(documents, data_dir_path, cnn_path)
    print(f"CNN heights {heights_cnn}")
    # get heights with binarization
    print("Get heights with Connected Components Analysis")
    _, heights_bin = heights_from_all_images(
        files_list=pd.Series(documents), dir_path=data_dir_path
    )
    print(f"bin heights {heights_bin}")
    # get heights with resnet18
    heights_resnet18 = get_heights_for_resnet18_resizing(documents, data_dir_path, resnet18_path)
    print(f"ResNet18 heights {heights_resnet18}")
    # save heights
    heights = {"binarization": heights_bin, "cnn": heights_cnn, "resnet18": heights_resnet18}

    # dictionaries to save training and validation images and corresponding scripts
    training_gt = {"baseline": {}, "binarization": {}, "cnn": {}, "resnet18": {}}
    validation_gt = {"baseline": {}, "binarization": {}, "cnn": {}, "resnet18": {}}

    for sample in tqdm(doc_script):
        fname = sample[0]
        main_group_script = sample[1]
        if main_group_script == "-":
            continue

        # if estimated height is 0 or nan then continue
        indx = documents.index(fname)
        if heights_cnn[indx] == 0:
            continue

        if heights_resnet18[indx] == 0:
            continue

        if np.isnan(heights_bin[indx]):
            continue

        if random() < 0.15:
            # resize validation image
            try:
                im = Image.open(data_dir_path / Path(f"./{fname}"))
            except UnidentifiedImageError:
                print(f"Could not open {fname}")
                continue

            for resizing_mode in ["baseline", "binarization", "cnn", "resnet18"]:
                new_im = im.copy()
                new_im, error = resizing_images(
                    new_im, resizing_mode, norm_height, heights, indx, fname
                )
                if error:
                    continue
                h = hashlib.md5()
                h.update(new_im.tobytes())
                cs = h.hexdigest()
                path = data_dir_path / Path(
                    f"./{resizing_mode}/typegroupdataset-validation/{main_group_script}/{cs}.jpg"
                )

                new_im.convert('1').save(path, "JPEG", quality=100)
                validation_gt[resizing_mode][path] = sample[1:]
                with open(
                    data_dir_path / Path(f"./{resizing_mode}/training-gt.map"), "wb"
                ) as f:
                    pickle.dump(training_gt[resizing_mode], f)
                    pickle.dump(validation_gt[resizing_mode], f)
                continue

        for resizing_mode in ["baseline", "binarization", "cnn", "resnet18"]:
            try:
                im = Image.open(data_dir_path / Path(f"./{fname}"))
            except UnidentifiedImageError:
                print(f"Could not open {fname}")
                continue
            # resize image height such as the text height will be the norm_height
            new_im = im.copy()
            new_im, error = resizing_images(
                new_im, resizing_mode, norm_height, heights, indx, fname
            )
            if error:
                continue
            # check the new size of the image
            if new_im.size[0] < crop_size or new_im.size[1] < crop_size:
                print(f"Too small: {fname}")
                continue  # too small

            h = hashlib.md5()
            h.update(new_im.tobytes())
            cs = h.hexdigest()
            path = data_dir_path / Path(
                f"./{resizing_mode}/typegroupdataset-training/{main_group_script}/{cs}.jpg"
            )

            new_im.convert('1').save(path, "JPEG", quality=100)
            training_gt[resizing_mode][path] = sample[1:]
