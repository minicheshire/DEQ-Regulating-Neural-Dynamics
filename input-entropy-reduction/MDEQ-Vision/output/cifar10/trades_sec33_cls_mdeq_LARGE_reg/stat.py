import torch
import numpy as np
import json
import sys

TrainGrad  = sys.argv[1]
AttackGrad = sys.argv[2]
attack = sys.argv[3]

ansc = [[] for _ in range(9)]
ansp = [[] for _ in range(9)]
for i in range(105):
    now = torch.load("train-{}-attack-{}-VAL_DATA-{}-{}".format(TrainGrad, AttackGrad, attack, i))
    if i and i % 10 == 0:
        print("now {}".format(i))
    for j in range(9 if attack != "pgd-es-0" else 1):
        ansc[j].append(now["CLEAN"]["acc1"][j])#*now["CLEAN"]["x_adv"].shape[0])
        ansp[j].append(now["PERTURBED"]["acc1"][j])#*now["PERTURBED"]["x_adv"].shape[0])

json.dump(ansc, open("Train-{}-Attack-{}-{}-ansc_val.json".format(TrainGrad, AttackGrad, attack), "w"))
json.dump(ansp, open("Train-{}-Attack-{}-{}-ansp_val.json".format(TrainGrad, AttackGrad, attack), "w"))
