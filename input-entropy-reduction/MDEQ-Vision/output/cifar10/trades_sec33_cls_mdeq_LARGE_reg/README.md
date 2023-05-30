Place the trained checkpoint here. The name should be `unroll-checkpoint.pth.tar`.

The `stat.py` and `res.py` are the scripts to calculate the standard and robust accuracy at each state z_n under a single attack.

First run `python stat.py unroll [ATTACK_INTERM] [ATTACK]` to enumerate all saved data and get all the statistics:

The `[ATTACK_INTERM]` string is in the form of `[A]unroll-[B]-[C]`, where
[A] = 'F' or 'M', corresponding to \lambda = 1 or \lambda=0.5 in Eq. (11), respectively;
[B] = 0, 1, 2, 3, 4, 5, 6, or 7. This represents $i$ in Eq. (11), namely, which intermediate state is to be unrolled in the intermediate attack;
[C] = 1, 2, 3, 4, 5, 6, 7, 8, or 9. This represents $K_a$ in Eq. (11), namely, how many unrolling steps is used in the intermediate attack.

The `[ATTACK]` string represents the type of attack. [ATTACK] can be 'pgd', 'apgd-ce', 'apgd-t', 'fab-t', 'square'.

After processing with `stat.py`, two json files are outputed. Then run `python res.py unroll [ATTACK_INTERM] [ATTACK]` to get the accuracy results at each states.

Leveraging the early state defense in Yang et al. (2022), we use the accuracies given by the last but one state.
