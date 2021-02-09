"""Pandas Dataset class definition.

Includes the following subfunctions.
- define_feature: Define temporal, static, label, treatment, and time.
- train_val_test_split: Split the data into train, valid, and test sets.
- cv_split: Split the data into train, valid, and test sets for cross validation.
- get_fold: Get the dataset for a certain fold.
"""

# Necessary packages
import numpy as np
from sklearn.model_selection import KFold


class PandasDataset:
    """Define Pandas Dataset class for mixed type data (static + temporal).

    Attributes:
        - static_data: raw static data set.
        - temporal_data: raw temporal data set.
    """

    def __init__(self, static_data, temporal_data):
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.is_feature_defined = False
        self.is_validation_defined = False
        self.temporal_feature = None
        self.static_feature = None
        self.label = None
        self.treatment = None
        self.time = None
        self.feature_name = None
        self.sample_size = None
        self.n_fold = None
        self.fold_list = None
        self.problem = None
        self.label_name = None

    def define_feature(
        self, temporal_feature, static_feature, label, treatment, time, feature_name, problem, label_name
    ):
        """Define temporal, static, label, treatment, and time.
        
        Args:
            - temporal_feature: time-series features
            - static_feature: non time-series features
            - label: end-point to be predicted
            - treatment: possible treatment
            - time: measurement time
            - feature_name: the column names of the temporal and static features
            - problem: one-shot or online
            - label_name: the column name of label
        """
        self.is_feature_defined = True
        self.temporal_feature = temporal_feature
        self.static_feature = static_feature
        self.label = label
        self.treatment = treatment
        self.time = time
        self.feature_name = feature_name
        self.sample_size = len(self.temporal_feature)
        self.problem = problem
        self.label_name = label_name

    def train_val_test_split(self, prob_val, prob_test, seed=666):
        """Split the data into train, valid, and test sets.
        
        Args:
            - prob_val: the ratio of validation dataset
            - prob_test: the ratio of testing dataset
            - seed: random seed
        """
        # Feature should be defined before dividing train/valid/test datasets
        assert self.is_feature_defined
        np.random.seed(seed)
        # Determine the number of samples in each dataset
        n_val = int(self.sample_size * prob_val)
        n_test = int(self.sample_size * prob_test)
        n_train = self.sample_size - n_val - n_test

        # Determine the index of train/valid/test datasets
        all_idx = np.asarray([i for i in range(self.sample_size)])
        idx = np.random.permutation(self.sample_size)
        train_idx = idx[0:n_train]
        valid_idx = idx[n_train : (n_train + n_val)]
        test_idx = idx[(n_train + n_val) :]
        # Set the outputs
        self.fold_list = [{"train": train_idx, "val": valid_idx, "test": test_idx, "all": all_idx}]
        self.n_fold = 1
        self.is_validation_defined = True

    def cv_split(self, n_fold=5, prob_val=0.2, seed=666):
        """Split the data into train, valid, and test sets for cross validation.
        
        Args:
            - n_fold: the number of fold for cross validation
            - prob_val: the ratio of validation dataset
            - seed: random seed
        """
        # Feature should be defined before dividing train/valid/test datasets
        assert self.is_feature_defined

        # Define K-fold Cross validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        x = list(range(self.sample_size))
        self.fold_list = []
        for train, test in kf.split(x):
            if prob_val > 0:
                n_val = int(len(train) * prob_val)
                np.random.shuffle(train)
                val_in_train = train[:n_val]
                rest_train = train[n_val:]
            else:
                rest_train = train
                val_in_train = None
            # Set the outputs
            fold_dict = {"train": rest_train, "val": val_in_train, "test": test}
            self.fold_list.append(fold_dict)
            self.n_fold = n_fold
            self.is_validation_defined = True

    def get_fold(self, fold, split):
        """Get the dataset for a certain fold.
        
        Args:
            - fold: fold index
            - split: split setting (should be among 'train', 'val', 'test', 'all')
            
        Returns:
            - x: temporal features
            - s: static features
            - lab: labels
            - t: time
            - treat: treatments
        """
        if not self.is_validation_defined:
            return self.temporal_feature, self.static_feature, self.label, self.time, self.treatment
        else:
            assert split in ("train", "val", "test", "all")
            assert fold <= self.n_fold
            inds = self.fold_list[fold][split]
            if inds is None or len(inds) == 0:
                print("The requested split has length 0. Returning Nones.")
                return None, None, None, None, None
            # Returns the following 5 outputs
            x = PandasDataset._safe_slice(self.temporal_feature, inds)
            s = PandasDataset._safe_slice(self.static_feature, inds)
            lab = PandasDataset._safe_slice(self.label, inds)
            t = PandasDataset._safe_slice(self.time, inds)
            treat = PandasDataset._safe_slice(self.treatment, inds)

            return x, s, lab, t, treat

    def get_bo_data(self):
        # todo: get required data set in bo format
        pass

    @staticmethod
    def _safe_slice(array, inds):
        if array is None:
            return None

        if len(array) == 0:
            return None

        return array[inds, ...]
