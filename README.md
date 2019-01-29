## T-GRAANK
A python implementation of the <i>Temporal-GRAdual rANKing</i> algorithm.<br>
<!-- Research paper published at FuzzIEEE 2019 International Conference on Fuzzy Systems (New Orleans): link<br> -->

### Getting Started:
Example Python program (file: Main.py)<br>
```python
from python.algorithm.UserMain import *
main("data/test.csv", 0, 0.1, 0.98)
```

Output:
```
Dataset Ok
{'Transformation': 'n+1', 'Representativity': 0.98, 'Included Rows': 49, 'Total Rows': 50}
1 : exercise_hours**
2 : stress_level
Pattern : Support
{'1+', '2+'} : 0.2814625850340136 | ~ +2.0 days : 1.0
{'2-', '1+'} : 0.1870748299319728 | ~ +1.4 days : 1.0
---------------------------------------------------------
```

The input parameters are: ```main(fileName.csv, referenceItem, minimumSupport, minimumRepresentativity)```. You are required to use a <strong>file</strong> in csv format and make sure the <i>timestamp column</i> is the first column in the file. You specify:
* <strong>reference item</strong> - column\attribute that is the base of the temporal transformations
* <strong>minimum support</strong> - the threshold count of frequent FtGPs
* <strong>mimimum representativity item</strong> - the threshold count of transformations to be performed on the data-set

### Credits:
1. Prof. Anne Laurent - LIRMM <i>Université de Montpellier 2</i><br>
2. Dr. Joseph Orero - Faculty of IT, <i>Strathmore University<br>
3. Office of the <strong>Co-operation and Cultural Service</strong>, Embassy of France in Kenya<br>
  
### License:
MIT
