""" This script is written following the example of creating a MLP regression model with PyTorch in
https://www.machinecurve.com/index.php/2021/07/20/how-to-create-a-neural-network-for-regression-with-pytorch/ """

from pathlib import Path
import torch as t
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from src.text_height_estimation.data_preprocessing import get_train_test_data
import matplotlib.pyplot as plt


class CNN(t.nn.Module):
    """
    Convolutional Neural Network for regression.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = t.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(2, 2)
        )
        self.conv2 = t.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(2, 2)
        )
        self.conv3 = t.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(2, 2)
        )
        self.relu = t.nn.ReLU()
        self.pool1 = t.nn.MaxPool2d(2)
        self.pool2 = t.nn.AdaptiveAvgPool2d((15, 15))
        self.fc1 = t.nn.Linear(15 * 15 * 1, 32)
        self.fc2 = t.nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 15 * 15 * 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_model(trainloader: DataLoader, model: CNN) -> CNN:
    """
    Train the model with the given data from the trainloader.
    :param trainloader: a DataLoader containing the training dataset.
    :param model: a convolutional neural network model.
    :return: a convolutional neural network model trained to estimate the text height in a document image.
    """
    # Loss function and optimizer definition
    loss_function = t.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # store the loss value
    loss_list = []

    # Training loop
    for epoch in range(0, 500):

        print(f"Starting epoch {epoch}")

        # Setting the current loss value
        current_loss = 0.0
        # Set on training mode
        model.train()
        # Iteration over the DataLoader for training data
        for i, data in enumerate(trainloader):
            # input_documents and target_heights preperation
            input_documents, target_heights = data
            target_heights = target_heights.view(-1, 1)
            # Zero the gradients
            optimizer.zero_grad()

            # Performing forward pass
            outputs = model(input_documents)

            # Computing loss
            loss = loss_function(outputs, target_heights)

            # Performing backward pass
            loss.backward()

            # Optimization
            optimizer.step()

            # Adding the found loss to the loss value for the current epoch
            current_loss += loss.item()

            # Printing statistics after every 10th batch
            if (i + 1) % 9 == 0:

                # Storing current_loss
                loss_list.append(current_loss)
                print(f"Loss after mini-batch {i + 1}: {current_loss }")
                current_loss = 0.0
        # check if the loss value is steady for 10 consequetive epochs, break the training loop
        if loss_list[epoch] <= 15:
            steady = True
            for i in range(10):
                # compare the loss in the 10 last epochs
                test = (
                    abs(loss_list[epoch] - loss_list[epoch - i])
                    <= 0.1 * loss_list[epoch]
                )
                steady = steady & test
            if steady:
                break

    # Process is complete.
    print("Training process has finished.")
    # plot the loss value
    plt.rcParams["font.family"] = "serif"
    plt.plot(list(range(1, len(loss_list) + 1)), loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss with respect to epochs")
    plt.show()
    return model


def evaluate_model(testloader: DataLoader, model: CNN):
    """
    Evaluate the performance of the model with data different from the training data.
    :param testloader: a DataLoader containing the testing dataset.
    :param model: a convolutional neural network model trained to estimate the text height in a document image.
    :return: print the accuracy of the model. A correctly predicted height
    """
    # Model evaluation
    correct = 0
    total = 0
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with t.no_grad():
        for _, data in enumerate(testloader):
            input_documents, target_heights = data

            # calculate outputs by running input_documents through the network
            predicted_heights = model(input_documents)
            for j, value in enumerate(predicted_heights):
                # get absolute value of the difference betwee the target and predicted value
                absolute_diff = abs(target_heights[j] - value)
                if absolute_diff <= target_heights[j] * 0.2:
                    correct += 1
            # update total value of the input documents
            total += input_documents.shape[0]
    # Printing statistics
    print(
        f"Accuracy of the network on the {total} test input_documents: {100 * correct / total}%"
    )
    # Process is complete.
    print("Testing process has finished.")


# Paths definition
REPO_PATH = str(Path(__file__).parent.parent.parent)
dir_path = REPO_PATH / Path("./data/labeled_data")
file_csv_path = dir_path / Path("./data_info_original.csv")


def main():
    # Data loading
    trainloader, testloader = get_train_test_data(
        file_csv_path=file_csv_path, dir_path=dir_path
    )
    # Model initialization
    cnn = CNN()
    # Moddel training
    trained_cnn = train_model(trainloader, cnn)
    # Model evaluation
    evaluate_model(testloader, trained_cnn)
    # Model saving
    file_path = REPO_PATH / Path("./data/results/height_estimation_cnn.the")
    t.save(trained_cnn.state_dict(), file_path)


if __name__ == "__main__":
    main()
