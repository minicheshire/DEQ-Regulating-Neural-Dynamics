# Input Entropy Reduction (the Sec. 3.2 method)

Input entropy reduction is an inference-time defense technique, which progressively updates the input along the neural dynamics of DEQ models. The core implementation lies at Lines 462~516 in `./MDEQ-Vision/lib/models/mdeq_core.py`. To run the codes, first place the trained checkpoint (by PGD-AT Base, PGD-AT + Sec.3.3, TRADES Base, or TRADES + Sec.3.3) at `./MDEQ-Vision/output/cifar10/{pgd_base/pgd_sec33/trades_base/trades_sec33}_cls_mdeq_LARGE_reg/unroll-checkpoint.pth.tar`. Then the command lines for running experiments with the Sec. 3.2 method are listed as follows (The results without the Sec.3.2 method are obtained with [the original Yang et al. (2022) repo](https://github.com/minicheshire/DEQ-White-Box-Robustness/tree/main/deq-evaluation-with-unrolled-intermediates)):

PGD-AT + Sec.3.2: 
`CUDA_VISIBLE_DEVICES=$1 python tools/cls_valid_observe_train.py --cfg experiments/cifar/pgd_base_cls_mdeq_LARGE_reg.yaml --TrainGrad unroll --AttackGrad [C] --attack [D]`

PGD-AT + Sec.3.2 + Sec.3.3:
`CUDA_VISIBLE_DEVICES=$1 python tools/cls_valid_observe_train.py --cfg experiments/cifar/pgd_sec33_cls_mdeq_LARGE_reg.yaml --TrainGrad unroll --AttackGrad [C] --attack [D]`

TRADES + Sec.3.2:
`CUDA_VISIBLE_DEVICES=$1 python tools/cls_valid_observe_train.py --cfg experiments/cifar/trades_base_cls_mdeq_LARGE_reg.yaml --TrainGrad unroll --AttackGrad [C] --attack [D]`

TRADES + Sec.3.2 + Sec.3.3:
`CUDA_VISIBLE_DEVICES=$1 python tools/cls_valid_observe_train.py --cfg experiments/cifar/trades_sec33_cls_mdeq_LARGE_reg.yaml --TrainGrad unroll --AttackGrad [C] --attack [D]`

[C] demonstrate the configurations of the unrolled intermediate state attack, and [D] represents the type of attack technique. The formats of [C] and [D] follow Yang et al. (2022) ([this link](https://github.com/minicheshire/DEQ-White-Box-Robustness/blob/main/deq-evaluation-with-unrolled-intermediates/README.md)):

[C] = [C1]unroll-[C2]-[C3], where 
- [C1] = "M" when lambda = 0.5, and "F" when lambda = 1 in Eq. (11);
- [C2] = 0, 1, 2, 3, 4, 5, 6, or 7, which indicates the [C2]-th intermediate state to be unrolled (the $i$ in Eq. (11)); 
- [C3] = 1, 2, 3, 4, 5, 6, 7, 8, or 9, which indicates the unrolling steps $K_a$ in Eq. (11).

[D] = 'pgd', 'apgd-ce', 'apgd-t', 'fab-t', or 'square'. The specific type of attack technique.

After running the command, you should find the saved files for all mini-batches in the `./MDEQ-Vision/output/cifar10/{pgd_base/pgd_sec33/trades_base/trades_sec33}_cls_mdeq_LARGE_reg/` directory. We've provided two scripts `stat.py` and `res.py` for the calculation of the standard and robust acc. at each state z_n -- check the [README.md]() for details.
