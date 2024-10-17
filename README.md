## GATLM
Geometric Analysis of Transformer Time Series Forecasting Latent Manifolds

# Our main results:
-Transformer forecasting manifolds exhibit two phases—dimensionality and curvature drop or remain fixed during encoding, then increase during decoding. 

<div align=center><img src="figures/.png" width="70%"></div>

-This behavior is consistent across architectures and datasets.

-The MAPC estimate correlates with test mean squared error, enabling model comparison without the test set.
<div align=center><img src="figures/.png" width="70%"></div>

-Geometric properties of the manifolds stabilize within a few training epochs.

<div align=center><img src="figures/.png" width="70%"></div>
  
## Training

In the repository, you can find a training script for Autoformer and FEDformer on the following datasets: ETTm1, ETTh1, ETTm2, ETTh2, Electricity, Traffic and Weather.
To run the training process run the following command:
```
python train.py 
```
You can train all the models by running the following shell code separately:

```

```

## Intrinsic Dimension and Curvature evaluation
To estimate the intrinsic dimension and curvature of the latent representations, execute the following command:
```
python est_curv.py 
```


## Paper
```

```
