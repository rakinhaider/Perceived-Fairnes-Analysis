import unittest

import numpy as np
import pandas as pd

from utils import aggregate_response, merge_cd_columns
from heatmap import get_heatmap_data


class TestHeatmap(unittest.TestCase):
    def test_merged_cd_columns(self):
        data_dirs = ['10312021']
        fnames = ['10312021']
        fnames = [f + '_approved.csv' for f in fnames]
        df = aggregate_response(data_dirs, fnames, root_dir='../')
        df = merge_cd_columns(df)

    def test_get_heatmap_data(self):
        data = [['High', 'Definitely model X'],
                ['High', 'Probably model X'],
                ['Moderate', 'Probably model Y'],
                ['Low', 'Probably model Y'],
                ['Low', 'Probably model Y'],
                ['Low', 'Definitely model Y']]
        data = pd.DataFrame(data, columns=['IFPI', 'Q10.20'])
        # result = [[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 2], [0, 0, 1]]
        result = np.array([[0.5, 0, 0], [0.5, 0, 0], [0, 0, 0],
                           [0, 1, 0.66666667], [0, 0, 0.33333333]])
        heatmap_data = get_heatmap_data(data, 'IFPI', 'Q10.20')
        assert np.allclose(heatmap_data, result)

