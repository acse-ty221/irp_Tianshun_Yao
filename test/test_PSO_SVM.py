from pytest import fixture
import pandas as pd
import numpy as np
import os
import sys

BASE_PATH = os.path.dirname(__file__)
parent_dir = os.path.dirname(BASE_PATH)
sys.path.append(parent_dir)

from tools.DataPre import DetectOutliers
from tools import gol


@fixture(scope='module')
def tools():
    import tools
    return tools


def test_import(tools):
    assert tools


@fixture(scope='module')
def pso(tools):
    NGEN = 60
    popsize = 30
    low = [0.001, 0.001, 0.01]
    up = [1000, 1000, 8]
    parameters = [NGEN, popsize, low, up]
    feature_test = np.ones((20, 5))
    targets_test = np.ones((20, 1))
    return tools.PSOSVM.PSO(parameters,
                            feature_test, targets_test)


def test_attributes(pso):
    for key in ('num_gen', 'pop_size', 'num_var', 'bound',
                'pop_x', 'pop_v', 'p_best', 'g_best'):
        assert hasattr(pso, key)


def test_outlierDetect():
    global outlier_test
    outlier_test = np.ones(21)
    outlier_test[20] = 100
    outlier_test_df = pd.DataFrame(outlier_test, columns=["targets"])
    outlier_test_fea = np.ones(21)
    outlier_test_fea_df = pd.DataFrame(outlier_test_fea, columns=["fea"])
    test_new_targets = outlier_test_df
    test_feature_transformed = outlier_test_fea_df
    gol._init()
    IO = DetectOutliers(
        feature=test_feature_transformed,
        targets=test_new_targets)
    test_new_targets = gol.get_value("new_targets")
    assert isinstance(IO, int)
    assert IO == 1
    assert np.mean(np.array(test_new_targets).ravel()) == 1
