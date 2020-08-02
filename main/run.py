import itertools

import pandas as pd
import numpy as np
from preprocess.run import OUTPUT_PATH
from main.dataset import LunaDataSet

VAL_PCT = 0.2

meta = pd.read_csv(f'{OUTPUT_PATH}/meta.csv', index_col=0).sample(frac=1).reset_index(drop=True)
meta_group_by_series = meta.groupby(['seriesuid']).indices
list_of_groups = [{i: list(meta_group_by_series[i])} for i in meta_group_by_series.keys()]
np.random.shuffle(list_of_groups)
val_split = int(VAL_PCT * len(list_of_groups))
val_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[:val_split]]))
train_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[val_split:]]))
lds = LunaDataSet(train_indices, meta)
