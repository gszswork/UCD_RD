import torch 
import argparse
from dataset import UCDDataset
from process import get_dataset
from model import *
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    datasetname = 'Twitter'
    in_dataset, out_dataset = get_dataset(datasetname)
