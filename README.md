## T-GRAANK
A python implementation of the <i>Temporal-GRAdual rANKing</i> algorithm. The algorithm extends the <i>GRAANK</i> algorithm to the case of temporal gradual tendencies.<br>
<!-- Research paper published at FuzzIEEE 2019 International Conference on Fuzzy Systems (New Orleans): link<br> -->

### List of Files:
1. README.md (this file)
2. main.py
3. run_tgraank.py
4. tgraank.py
5. fuzzy_temporal.py
6. data_transform.py
7. test.csv

### Getting Started:
Example Python program (file: main.py)<br>
```python
from python.algorithm.run_tgraank import *
main("data/test.csv", 0, 0.5, 0.5)
```

Output:
```
Dataset Ok
{'Transformation': 'n+3', 'Representativity': 0.94, 'Included Rows': 47, 'Total Rows': 50}
1 : exercise_hours**
2 : stress_level
Pattern : Support
{'1+', '2+'} : 0.5060129509713228 | ~ +6.0 days : 1.0
-------------------------------------------------------------------------------------------
# can be interpreted as: the more exercise_hours, the more stress_level almost 6 days later
```

The input parameters are: ```main(fileName.csv, referenceItem, minimumSupport, minimumRepresentativity)```. You are required to use a <strong>file</strong> in csv format and make sure the <i>timestamp column</i> is the first column in the file. You specify:
* <strong>reference item</strong> - column\attribute that is the base of the temporal transformations
* <strong>minimum support</strong> - threshold count of frequent FtGPs
* <strong>mimimum representativity item</strong> - threshold count of transformations to be performed on the data-set
  
### License:
* MIT

### Reference:
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09), Troels Andreasen, Ronald R. Yager, Henrik Bulskov, Henning Christiansen, and Henrik Legind Larsen (Eds.). Springer-Verlag, Berlin, Heidelberg, 382-393.
