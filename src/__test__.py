import numpy as np
from TemporalGP.TGP.tgrad_ami import TGradAMI


# f_path = "../../datasets/rain_temp2013-2015.csv"
f_path = "../../datasets/air_quality25.csv"
# f_path = "../../datasets/air_quality1k.csv"

tgp = TGradAMI(f_path, False, 0.5, 2, 1, 1)
# tgp.discover_tgp()

a = 2
b = 4
c = 6
x = np.array([1, 2, 3, 4, 8])

print(f"a: {a}, b: {b}, c: {c}")
print(f"x-data: {x}\n")
TGradAMI.learn_best_mf_w_ml(a, b, c, x)
