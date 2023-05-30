import pandas as pd
import numpy as np

def output_df(z: list, alpha: np.ndarray, tau: np.ndarray):
    for idx, class_i in enumerate(z):
        