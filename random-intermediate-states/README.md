# Random Intermediate States (the Sec. 3.3 method)

The random intermediate states method is a training-time technique, which randomly uses some intermediate state to calculate for the adversarial loss. The core implementation lies at Line 440 of `./MDEQ-Vision/lib/models/mdeq_core.py`; switching Line 440 to using the final state recovers the baseline version.

In this repo, we provide the code for training DEQ models with TRADES. To train a DEQ model with TRADES + Sec.3.3 method, run

`CUDA_VISIBLE_DEVICES=$1 python tools/cls_train.py --cfg experiments/cifar/trades_sec33_cls_mdeq_LARGE_reg.yaml`


