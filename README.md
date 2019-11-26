[![Build Status](https://travis-ci.org/owuordickson/t-graank.svg?branch=master)](https://travis-ci.org/owuordickson/t-graank)
## T-GRAANK
A python implementation of the <i>Temporal-GRAdual rANKing</i> algorithm. The algorithm extends the <i>GRAANK</i> algorithm to the case of temporal gradual tendencies. Research paper appears in the proceedings of Fuzz-IEEE 2019 International Conference on Fuzzy Systems held at New Orleans, USA 23rd-26th June 2019: [doi](10.1109/FUZZ-IEEE.2019.8858883)<br>

### List of Files:
1. README.md (this file)
2. t_graank.py
3. fuzzy_temporal.py
4. data_transform.py
5. DATASET.csv

### Usage:
Use it a command line program with the local package:
```
$python t_graank.py -f fileName.csv -c refColumn -s minSupport  -r minRepresentativity
```

The input parameters are: ```fileName.csv, refColumn, minSupport, minRepresentativity```. You are required to use a <strong>file</strong> in csv format and make sure the <i>timestamp column</i> is the first column in the file. You specify:
* <strong>reference item</strong> - column\attribute that is the base of the temporal transformations
* <strong>minimum support</strong> - threshold count of frequent FtGPs
* <strong>mimimum representativity item</strong> - threshold count of transformations to be performed on the data-set

Example with a data-set and specified values<br>
```
$python t_graank.py -f DATASET.csv -c 0 -s 0.5 -r 0.5
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

### License:
* MIT

### Reference:
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09), Troels Andreasen, Ronald R. Yager, Henrik Bulskov, Henning Christiansen, and Henrik Legind Larsen (Eds.). Springer-Verlag, Berlin, Heidelberg, 382-393.
