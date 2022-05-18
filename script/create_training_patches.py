from pathlib import Path
from PIL import ImageFile
from src.script_recognition.training_patches_utils import (
    count_samples_per_typegroup,
    create_training_validation_directories,
    split_training_validation_sets,
    plot_scripts
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
REPO_PATH = str(Path(__file__).parent.parent)
data_dir_path = REPO_PATH / Path("./data/ICDAR2017_CLaMM_Training")
training_csv_path = data_dir_path / Path("./ICDAR2017_CLaMM_Training.csv")
cnn_path = REPO_PATH / Path("./data/results/height_estimation_cnn.the")
resnet18_path = REPO_PATH / Path("./data/results/height_estimation_resnet18.the")

# count samples per type group
count = count_samples_per_typegroup(training_csv_path=training_csv_path)
# plot_scripts(count)

# create training and validation directories
create_training_validation_directories(
    count=count,
    data_dir_path=data_dir_path,
)

# split the data into training and validation sets
split_training_validation_sets(
    training_csv_path=training_csv_path,
    data_dir_path=data_dir_path,
    cnn_path=cnn_path,
    resnet18_path=resnet18_path,
    norm_height=40,
)
