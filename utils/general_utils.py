import os
import logging
import contextlib
import random
from typing import Tuple, Optional
import multiprocessing

import numpy as np
import tensorflow as tf


def tf_set_log_level(level: int) -> None:
    if level >= logging.FATAL:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    if level >= logging.ERROR:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if level >= logging.WARNING:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logging.getLogger("tensorflow").setLevel(level)


def silence_tf() -> None:
    tf_set_log_level(logging.FATAL)


def _get_tf_logging_state() -> Tuple[int, str]:
    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        original_cpp_log_level = os.environ["TF_CPP_MIN_LOG_LEVEL"]
    else:
        original_cpp_log_level = None
    original_logging_verbosity = tf.compat.v1.logging.get_verbosity()
    return original_logging_verbosity, original_cpp_log_level


def _restore_tf_logging_state(original_logging_verbosity: int, original_cpp_log_level: str) -> None:
    if original_cpp_log_level is not None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = original_cpp_log_level
    else:
        if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
            del os.environ["TF_CPP_MIN_LOG_LEVEL"]
    tf.compat.v1.logging.set_verbosity(original_logging_verbosity)


@contextlib.contextmanager
def silence_tf_ctx():
    tf_logging_state = _get_tf_logging_state()
    silence_tf()
    try:
        yield
    finally:
        _restore_tf_logging_state(*tf_logging_state)


@contextlib.contextmanager
def tf_set_log_level_ctx(level: int):
    tf_logging_state = _get_tf_logging_state()
    tf_set_log_level(level)
    try:
        yield
    finally:
        _restore_tf_logging_state(*tf_logging_state)


def fix_all_random_seeds(random_seed):
    """
    Fix random seeds etc. for experiment reproducibility.
    
    Args:
        random_seed (int): Random seed to use.
    """
    # os.environ['PYTHONHASHSEED']=str(random_seed)  # May be needed.
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.compat.v1.random.set_random_seed(random_seed)


@contextlib.contextmanager
def tf_fixed_seed_session(seed):
    fix_all_random_seeds(seed)
    try:
        yield
    finally:
        tf.compat.v1.reset_default_graph()


def _get_cuda_visible_devices() -> Optional[str]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        value = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        value = None
    return value


def _reset_cuda_visible_devices(original_value: Optional[str]) -> None:
    if original_value is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_value


def tf_set_cuda_visible_devices(value: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = value


def tf_cuda_invisible(disable: bool = True) -> None:
    if disable:
        tf_set_cuda_visible_devices("-1")


@contextlib.contextmanager
def tf_set_cuda_visible_devices_ctx(value: str):
    original_value = _get_cuda_visible_devices()
    tf_set_cuda_visible_devices(value)
    try:
        yield
    finally:
        _reset_cuda_visible_devices(original_value)

@contextlib.contextmanager
def tf_cuda_invisible_ctx(make_invisible: bool = True):
    original_value = _get_cuda_visible_devices()
    tf_cuda_invisible(make_invisible)
    try:
        yield
    finally:
        _reset_cuda_visible_devices(original_value)

def run_in_own_process(func, *args, set_tf_cuda_invisible=False, **kwargs):
    # TODO: Better Exception handling (in case child process raises exception).
    def wrapper(return_dict, set_tf_cuda_invisible, *args, **kwargs):
        with tf_cuda_invisible_ctx(make_invisible=set_tf_cuda_invisible):
            returned_tuple = func(*args, **kwargs)
            return_dict["returned_tuple"] = returned_tuple
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=wrapper, name="run_independent_process", args=(return_dict, set_tf_cuda_invisible, *args), kwargs=kwargs)
    p.start()
    p.join()
    return return_dict["returned_tuple"]
