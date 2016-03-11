Learning to Rank
======

A simple implementation of algorithms of learning to rank.

## Requirements
[tqdm](https://github.com/noamraph/tqdm)

[matplotlib v1.5.1](http://matplotlib.org/)

[numpy v1.10.1](http://www.numpy.org/)

[scipy]()

[chainer v1.5.1](http://chainer.org/)

[scikit-learn](http://scikit-learn.org/stable/)

and some basic packages.


## RankNet
### Pairwise comparison of rank

The original paper is available at http://research.microsoft.com/en-us/um/people/cburges/papers/ICML_ranking.pdf.

### Usage

Import and initialize

```
import RankNet
Model = RankNet.RankNet()
```

Fitting (automatically do training and validation)

```
Model.fit(X, y)
```

Possible options and defaults:

```
batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="RankNet.model"
```

## ListNet

### Listwise comparison of rank

The original paper is available at http://research.microsoft.com/en-us/people/tyliu/listnet.pdf.

NOTICE:
    The top-k probability is not written.
    This is listwise approach with neuralnets, 
    comparing two arrays by Jensen-Shannon divergence.


### Usage

Import and initialize

```
import ListNet
Model = ListNet.ListNet()
```

Fitting (automatically do training and validation)

```
Model.fit(X, y)
```

Possible options and defaults:

```
batchsize=100, n_epoch=200, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="ListNet.model"
```


## Author

If you have troubles or questions, please contact [shiba24](https://github.com/shiba24).

March, 2016

