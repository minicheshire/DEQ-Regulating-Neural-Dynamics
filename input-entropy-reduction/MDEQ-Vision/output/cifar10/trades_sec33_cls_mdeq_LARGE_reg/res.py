import json
import sys

TrainGrad  = sys.argv[1]
AttackGrad = sys.argv[2]
attack = sys.argv[3]

try:
    ansc_valid = json.load(open("Train-{}-Attack-{}-{}-ansc_val.json".format(TrainGrad, AttackGrad, attack), "r"))
    ansp_valid = json.load(open("Train-{}-Attack-{}-{}-ansp_val.json".format(TrainGrad, AttackGrad, attack), "r"))

    meann_c_valid = []
    meann_p_valid = []
    for jj in range(9 if attack != "pgd-es-0" else 1):
        meann_c_valid.append((sum(ansc_valid[jj][:-1])*96 + ansc_valid[jj][-1]*16)/10000)
        meann_p_valid.append((sum(ansp_valid[jj][:-1])*96 + ansp_valid[jj][-1]*16)/10000)

    print("Train-{}-Attack-{}".format(TrainGrad, AttackGrad))
    print("\t".join(["CLEAN"] + [str(round(_, 2)) for _ in meann_c_valid]))
    print("\t".join([attack] + [str(round(_, 2)) for _ in meann_p_valid]))

except:
    print(f"======={AttackGrad} Not Found=======")

