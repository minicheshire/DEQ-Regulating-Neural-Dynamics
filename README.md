# Improving Adversarial Robustness of Deep Equilibrium Models with Explicit Regulations Along the Neural Dynamics

This repo contains the source code for our ICML 2023 paper: [Improving Adversarial Robustness of Deep Equilibrium Models with Explicit Regulations Along the Neural Dynamics](). 

Our main technical contributions are input entropy reduction (Sec. 3.2) and random intermediate states (Sec. 3.3). The former is an inference-time defense strategy, which progressively updates the input along the neural dynamics of DEQ models to improve adversarial robustness. The latter is a training-time defense method, which randomly uses some intermediate state for the calculation of the adversarial loss during training. The two directories in this repo contains the implementations of the two methods. A large amount of the codes are inherited from the [original DEQ repo](https://github.com/locuslab/deq/tree/master/MDEQ-Vision) and the implementation of its [white-box robustness evaluation protocol](https://github.com/minicheshire/DEQ-White-Box-Robustness).

Consider citing our works if you find this repository useful:

1. A closer Look at the Adversarial Robustness of Deep Equilibrium Models

```
@inproceedings{
    yang2022a,
    title={A Closer Look at the Adversarial Robustness of Deep Equilibrium Models},
    author={Yang, Zonghan and Pang, Tianyu and Liu, Yang},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=_WHs1ruFKTD}
}

```

2. Improving Adversarial Robustness of DEQs with Explicit Regulations Along the Neural Dynamics

```
@inproceedings{
    yang2023improving,
    title={Improving Adversarial Robustness of DEQs with Explicit Regulations Along the Neural Dynamics},
    author={Yang, Zonghan and Li, Peng and Pang, Tianyu and Liu, Yang},
    booktitle={Proceedings of the 40th International Conference on Machine Learning},
    year={2023},
}
```
