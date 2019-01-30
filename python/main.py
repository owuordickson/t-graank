from python.algorithm.run_tgraank import main

import time
start_time = time.time()
main("data/test.csv", 0, 0.5, 0.5)
#main("data/rain_temp2013-2015_full.csv",0,0.5,0.5)
#main("data/ndvi_test.csv",0,0.5,0.8)
#main("data/ndvi_kenya.csv", 2, 0.5, 0.8)
#main("data/ndvi_towns_full.csv", 0, 0.5, 0.5)
print("--- %s seconds ---" % (time.time() - start_time))