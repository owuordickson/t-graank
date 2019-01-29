## T-GRAANK
A python implementation of the <i>Temporal-GRAdual rANKing</i> algorithm. The algorithm extends the <i>GRAANK</i> algorithm to the case of temporal gradual tendencies.<br>
<!-- Research paper published at FuzzIEEE 2019 International Conference on Fuzzy Systems (New Orleans): link<br> -->

### Getting Started:
Example Python program (file: Main.py)<br>
```python
from python.algorithm.UserMain import *
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
# interpreted as: the more exercise_hours, the more stress_level almost 6 days later
```

The input parameters are: ```main(fileName.csv, referenceItem, minimumSupport, minimumRepresentativity)```. You are required to use a <strong>file</strong> in csv format and make sure the <i>timestamp column</i> is the first column in the file. You specify:
* <strong>reference item</strong> - column\attribute that is the base of the temporal transformations
* <strong>minimum support</strong> - threshold count of frequent FtGPs
* <strong>mimimum representativity item</strong> - threshold count of transformations to be performed on the data-set

### Credits:
1. Prof. Anne Laurent - LIRMM, <i>Université de Montpellier</i>
2. Dr. Joseph Orero - Faculty of IT, <i>Strathmore University</i>
3. Office of the <strong>Co-operation and Cultural Service</strong>, Embassy of France in Kenya
  
### License:
* MIT
