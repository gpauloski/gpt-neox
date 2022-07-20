# Training with K-FAC

To enable K-FAC training, simply include the K-FAC training config ([configs/perlmutter/kfac.yml](kfac.yml)).
Many K-FAC parameters can be modified in the config file.
```
python ./deepy.py train.py [path/to/config.yml] ... configs/perlmutter/kfac.yml
```

E.g., to train GPT-XL on perlmutter, the following will work.
```
python ./deepy.py train.py configs/perlmutter/XL.yml configs/perlmutter/slurm.yml configs/perlmutter/kfac-XL.yml
```

Two KFAC configs are provided, `kfac-XL.yml` and `kfac-20B.yml` for the XL and 20B models, respectively.
The key changes between the two are that for the 20B model, tensor fusion is not used for communication to reduce memory costs and factors are saved to a separate checkpoint directory because of how large they are (rather than being saved in the model checkpoint file).

The setup instructions are the same as in the [README](../../README.md).
The only change is that [kfac-pytorch](https://github.com/gpauloski/kfac-pytorch) needs to be installed as well.

The most important config option to check is the `skip_layers`.
K-FAC cannot efficiently optimize the embedding layer or the final linear layer so these should be excluded.
However, the name of the final linear layer changes depending on the size of the model so if you get OOM errors, check that the final linear layer's name is correctly included in the list.
A regular expression that should catch the final layer has already been included.

If you want to change additional K-FAC parameters not provided in the example config, the K-FAC preconditioner is initialized in [megatron/training.py](megatron/training.py).
