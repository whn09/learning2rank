import numpy as np
import random

from rank import RankNet

n = 1000
d = 200
X = np.random.rand(n,d)
y = np.random.rand(n)

model = RankNet.RankNet()
random.seed(1313)
model.fit(X, y, 
          batchsize=16,
          n_iter=1000,
          n_units1=256,
          n_units2=256,
          tv_ratio=0.67,
          optimizerAlgorithm="Adam",
          savefigName="ranknet_result.jpg",
          savemodelName="RankNet.model")
