
from TemporalGP.TGP.tgrad_ami import TGradAMI

f_path = "../../datasets/rain_temp2013-2015.csv"
tgp = TGradAMI(f_path, False, 0.5, 1, 1, 4)
print(f"MI: {tgp.initial_mutual_info} \n")

