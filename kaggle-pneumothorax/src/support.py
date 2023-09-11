import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

n=100
# [writer.add_scalar("loss",np.random.rand( ), i) for i in range(n) ]

writer.flush()
print ('finisnh')