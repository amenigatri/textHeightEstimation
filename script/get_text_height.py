import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.text_height_estimation.connected_components import heights_from_all_images
from src.text_height_estimation.xml_conversion import import_xml_session


REPO_PATH = str(Path(__file__).parent.parent)
dir_path = REPO_PATH / Path("./data/labeled_data")
dirs = os.listdir(dir_path)

# list of document images
documents = []
# list of correspnding heights
heights = []
for file in dirs:
    # get xml files
    if ".xml" not in file:
        continue
    file_path = dir_path / Path(file)
    xmldict = import_xml_session(file_path=file_path)
    documents.append(xmldict["filename"])
    heights.append(xmldict["heightInPixels"])

# create dataframe
data = pd.DataFrame({"document": documents, "height": heights})

# save informations in csv file
file_csv_path = dir_path / Path("./data_info_original.csv")
if not Path.exists(file_csv_path):
    with open(file_csv_path, "w+", encoding="utf-8"):
        data.to_csv(file_csv_path, index=False)
else:
    data.to_csv(file_csv_path, index=False)

# get text height through connected components analysis

avg_heights, medians = heights_from_all_images(
    files_list=data["document"], dir_path=dir_path
)
# save the average and median values of the text height estimated through the connected components analysis
df = pd.DataFrame(
    {
        "document": data["document"],
        "ground_truth": data["height"],
        "avg_height": avg_heights,
        "median": medians,
    }
)
# save informations in csv file
new_file_path = REPO_PATH / Path("./data/results/connected_components_results.csv")
if not Path.exists(new_file_path):
    with open(new_file_path, "w+", encoding="utf-8"):
        df.to_csv(new_file_path, index=False)
else:
    df.to_csv(new_file_path, index=False)

# plot the ground truth and median values
plt.rcParams["font.family"] = "serif"
plt.scatter(x=df["ground_truth"], y=df["median"])
plt.xlabel("ground truth")
plt.ylabel("median")
plt.title("Ground truth height form LabelImg tool vs. Median from Connected Components")
plt.show()

# plot the ground truth ang mean values
plt.rcParams["font.family"] = "serif"
plt.scatter(x=df["ground_truth"], y=df["avg_height"])
plt.xlabel("ground truth")
plt.ylabel("average height")
plt.title("Ground truth height form LabelImg tool vs. Mean from Connected Components")
plt.show()


# accuracy of connected components analysis
correct_median = 0
for i, real_value in enumerate(df["ground_truth"]):
    absolute_diff = abs(real_value - df.at[i, "median"])
    if absolute_diff <= real_value * 0.2:
        correct_median += 1
print(
    "median accuracy is ", (correct_median * 100) / len(data.index)
)  # 76.66666666666667

