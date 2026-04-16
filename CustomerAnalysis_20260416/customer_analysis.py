import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CustomerAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'Mall_Customers.csv'), encoding ='gbk', header=None, sep=',')

if __name__ == "__main__":
    path = 'path'
    anlys = CustomerAnalysis(path)