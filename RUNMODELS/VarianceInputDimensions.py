import time
from sklearn.preprocessing import MinMaxScaler
import sys
import os
BASE_PATH = os.path.dirname(__file__)
parent_dir = os.path.dirname(BASE_PATH)
sys.path.append(parent_dir)
from tools import gol
from tools import LoadDataset, DetectOutliers, DataScatter,\
    FeatureSelect
from tools.AnalysisTools import variance_different_inputs


if __name__ == '__main__':
    sheet_name = 2012
    print("\nBegin to Load Data!\n")
    df_blast_data, feature, targets = LoadDataset(sheet_name)
    new_targets = targets
    print("\nData preprocessing start!\n")

    gol._init()
    print(df_blast_data)
    gol.set_value("df_blast_data", df_blast_data)
    gol.set_value("feature", feature)
    gol.set_value("targets", targets)

    # outlier detection
    targets_no_outliers_IO = DetectOutliers(feature, targets)
    gol.get_value("new_targets")
    if (targets_no_outliers_IO):
        print("\n")
        print("\nBefore outlier process :")
        DataScatter(feature, targets)
        print("\nAfter outlier process :")
        DataScatter(feature, new_targets)
        targets = new_targets
        print("The targets has been updated!")
    else:
        targets = new_targets
        print(targets)
        print("The dataset stays unchanged!")

    # feature selection
    if (sheet_name == 2012):
        data_fea = df_blast_data.iloc[1:, 1: 9]
        data_tar = df_blast_data.iloc[1:, -2]
    elif (sheet_name == 2018):
        data_fea = df_blast_data.iloc[:, [14, -2]]
        data_tar = df_blast_data.iloc[:, [-4]]

    gol.set_value("data_fea", data_fea)
    gol.set_value("data_tar", data_tar)
    print(gol.get_value("feature"))
    top_features_indices = [0, 0, 0, 0]
    feature_indices_IO = FeatureSelect(data_fea, data_tar, n=5)
    top_features_indices = gol.get_value("top_features_indices")
    if (feature_indices_IO):
        new_feature = df_blast_data.iloc[1:, top_features_indices]
        feature = new_feature
        print("The feature has been updated!")
    else:
        print("The feature stay unchangded!")

    print("\nData normalization start!\n")
    scaler = MinMaxScaler()
    # train standardized object
    scaler.fit(feature)
    # transform dataset
    feature_transformed = scaler.transform(feature)

    print("\nData has been normalized!\n")

    time_start = time.time()
    gol.set_value("feature_transformed", feature_transformed)
    gol.set_value("targets", targets)

    print("variance of different input dimensions: ")
    variance_different_inputs(df_blast_data)
