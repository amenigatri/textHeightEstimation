from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import logging
from torchvision import transforms
from torch.utils.data import DataLoader


class TextDataset(torch.utils.data.Dataset):
    """
    Prepare the dataset for regression
    """

    def __init__(self, doc, h):
        """
        :param doc: the document image name
        :param h: the text height
        """
        self.doc, self.h = doc, h

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, i):
        return self.doc[i], self.h[i]


def load_document_images(files_list: list, dir_path: str) -> list:
    """
    Load the input images given by files list and transform them to tensors.
    :param files_list: a pandas series contaning the names of the document images.
    :param dir_path: the path of the directory containing the document images.
    :return: a tensor of tensors correspondeing to the images from the files list.
    """
    # define a tranformation
    data = []
    trans = transforms.Compose(
        [
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )
    for _, doc in enumerate(files_list):
        try:
            # Load the image
            img_path = dir_path / Path(doc)
            img_arr = Image.open(img_path)
            # Grayscaling
            gray_im = img_arr.convert('1')
            data.append(trans(gray_im))
        except ImportError as e:
            print(e)
        except Exception as e:
            logging.exception(e)
    return data


def get_train_test_data(file_csv_path: str, dir_path: str) -> (DataLoader, DataLoader):
    """
    Prepare the training and testing datasets.
    :param file_csv_path: the path of the file containg the names and heights of the document images that will be used
    for training and testing the model.
    :param dir_path: the path of the directory containing the document images.
    :return: a tuple of data loaders corresponding respectively to the training and testing datasets.
    """
    documents_info = pd.read_csv(file_csv_path)
    # split into train and test data
    x_train, x_test, y_train, y_test = split_train_test(
        documents_info=documents_info, test_size=0.3
    )
    # load training data
    x_train = load_document_images(files_list=x_train.values, dir_path=dir_path)
    y_train = torch.Tensor(y_train.values)
    training_dataset = TextDataset(x_train, y_train)
    trainloader = DataLoader(training_dataset, batch_size=20, shuffle=True)
    # load testing data
    x_test = load_document_images(files_list=x_test.values, dir_path=dir_path)
    y_test = torch.Tensor(y_test.values)
    testing_dataset = TextDataset(x_test, y_test)
    testloader = DataLoader(testing_dataset, batch_size=5)
    return trainloader, testloader


def split_train_test(
    documents_info: pd.DataFrame, test_size: float
) -> (pd.Series, pd.Series, pd.Series, pd.Series):

    # get min and max value of height
    min_height = documents_info["height"].min()
    max_height = documents_info["height"].max()

    # calculate bin size
    step = (max_height - min_height) / 3

    # get five groups of documents by height value
    groups = []
    for i in range(2):
        groups.append(
            documents_info[
                (documents_info["height"] >= min_height + i * step)
                & (documents_info["height"] < min_height + (i + 1) * step)
            ]
        )
    # add last group separately to include the max height
    groups.append(
        documents_info[
            (documents_info["height"] >= min_height + 2 * step)
            & (documents_info["height"] <= max_height)
        ]
    )
    # get the train and test datasets
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in range(3):
        # get testing data
        n_test = int(len(groups[i].index) * test_size)
        test_df = test_df.append(groups[i].tail(n_test))

        # get training data
        n_train = len(groups[i].index) - n_test
        train_df = train_df.append(groups[i].head(n_train))

    # get x_train, x_test, y_train, y_test
    x_train = train_df["document"]
    x_test = test_df["document"]
    y_train = train_df["height"]
    y_test = test_df["height"]

    return x_train, x_test, y_train, y_test
