import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from itertools import chain
import warnings
from enum import Enum
from typing import Dict, List, Optional, Union
import sys
import os
import json
import gc
import functools
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import optuna

from transformers import (
    BertConfig,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorForTokenClassification,
    SpecialTokensMixin,
    BatchEncoding,
    get_scheduler,
)
from transformers.utils import logging, to_py_obj

from datasets import load_from_disk

# local modules
from .data import preload_and_process_data, get_data_loader
from .model import GeneformerMultiTask
from .utils import save_model
from .optuna_utils import create_optuna_study
from .collators import DataCollatorForMultitaskCellClassification