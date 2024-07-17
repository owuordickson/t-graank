import numpy as np
from TemporalGP.TGP.tgrad_ami import TGradAMI


# f_path = "../../datasets/rain_temp2013-2015.csv"
f_path = "../../datasets/air_quality25.csv"

tgp = TGradAMI(f_path, False, 0.5, 2, 1, 1)
tgp.discover_tgp()

squared_diff = (tgp.mi_arr - tgp.initial_mutual_info) ** 2
absolute_error = np.sqrt(squared_diff)

# print(f"Initial MI: {tgp.initial_mutual_info}\n")
# print(f"Delayed MIs: {tgp.mi_arr}\n")
# print(f"Abs.E.: {absolute_error}\n")
print(f"Optimal Step: {np.argmin(absolute_error, axis=0)}\n")
