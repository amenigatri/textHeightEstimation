from pathlib import Path
from torchvision.datasets import ImageFolder
import pickle
from src.script_recognition.evaluate_model import *


REPO_PATH = str(Path(__file__).parent)
data_dir_path = REPO_PATH / Path("./data/fontgroupsdataset-labels")
testing_csv_path = data_dir_path / Path("./labels-test.csv")


for resizing_mode in ["baseline", "binarization", "cnn"]:
    for net in ["densenet121", "resnet50"]:
        print(f"Evaluation of {net} for {resizing_mode} algorithm")
        # load model
        model = load_trained_model(net, resizing_mode)

        # prepare data
        testfolder_path = data_dir_path / Path(
            f"./{resizing_mode}/typegroupdataset-testing"
        )
        testing = ImageFolder(testfolder_path, transform=None)
        testing.target_transform = model.classMap.get_target_transform(
            testing.class_to_idx
        )

        testing_map_path = data_dir_path / Path(f"./{resizing_mode}/testing-gt.map")
        with open(testing_map_path, "rb") as f:
            testing_data = pickle.load(f)

        evaluate_model(model=model, testing_data=testing_data)
