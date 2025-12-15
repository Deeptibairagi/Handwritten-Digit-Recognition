import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import datasets,transforms
from torchview import draw_graph
from torch.utils.data import DataLoader,random_split
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from flask import Flask, render_template, request, jsonify
import io
from PIL import Image, ImageOps
import base64
import numpy as np

