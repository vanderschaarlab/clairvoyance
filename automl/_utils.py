"""Utilities for AutoML.
"""

# Necessary packages
import random
import time
from copy import deepcopy
import numpy as np


def error_function(BO_output):
    rmse1 = np.min(np.mean(BO_output, axis=1), axis=0)
    rmse2 = np.mean(np.min(BO_output, axis=0))
    return rmse1, rmse2


def get_opt_domain(domain):

    dim = len(domain)

    bounds = []
    bounds_type = []
    for i_domain in domain:
        bounds.append([i_domain["domain"][0], i_domain["domain"][-1]])
        bounds_type.append(i_domain["type"])

    bb = [bounds, bounds_type]
    return domain, dim, bb


def init_random_uniform(domain, n_points=25000, initial=False):

    result = []

    for k in range(int(n_points)):

        if initial:
            random.seed(k)
        else:
            random.seed(time.time())

        list_i = []
        for i_domain in domain:

            if i_domain["type"] == "continuous":
                kk = float(random.uniform(i_domain["domain"][0], i_domain["domain"][1]))
                list_i.append(kk)
            else:
                list_i.append(int(random.sample(i_domain["domain"], 1)[0]))

        result.append(list_i)
    return result


def min_list(obs):

    obs = -obs[:, -1]
    leng = len(obs)
    list = []
    a = obs[0]
    list.append(a)
    for i in range(1, leng):
        if obs[i] <= a:
            list.append(obs[i])
            a = deepcopy(obs[i])
        else:
            list.append(a)
    return list


def model_eval(dataset, model, metric):
    assert dataset.is_validation_defined
    model.fit(dataset)
    test_y_hat = model.predict(dataset)
    return metric.eval(dataset, test_y_hat)
