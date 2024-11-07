[![Build Status](https://travis-ci.org/owuordickson/t-graank.svg?branch=main)](https://travis-ci.org/owuordickson/t-graank)
## T-GRAANK
A python implementation of the <i>Temporal-GRAdual rANKing</i> algorithm. The algorithm extends the <i>GRAANK</i> algorithm to the case of temporal gradual tendencies. We have optimized the implementation of the algorithm by: (1) using Numpy functions for operations that are time-consuming and, (2) allowed parallel multiprocessing. The research paper is available via:

* D. Owuor, A. Laurent and J. Orero, "Mining Fuzzy-Temporal Gradual Patterns," 2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), 2019, pp. 1-6, doi: 10.1109/FUZZ-IEEE.2019.8858883.

### List of (important) Files:
1. README.md (this file)
2. cli_main.py (for running algorithm manually)
3. DATASET.csv (a dataset for quick testing)

### Installation

* Install Python version 3.11 on your computer.
* Download, extract the ```source code``` folder named **'t-graank'** and save it to your preferred location on your PC.
* Open a terminal application such as CMD. 
* Navigate to the location where you saved the **'t-graank'** folder using the terminal. 
* Execute the following commands:

```chatinput
cd t-graank
pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

### Usage:
Use it a command line program with the local package:

```
__main__.py -f datasets/DATASET.csv -t targetColumn -s minSupport  -r minRepresentativity -p allowMultiprocessing -c numCPUs
```

The input parameters are: ```fileName.csv, targetColumn, minSupport, minRepresentativity```. You are required to use a <strong>file</strong> in csv format and make sure the <i>timestamp column</i> is the first column in the file. You specify:
* <strong>target column</strong> - (int) column\attribute that is the base of the temporal transformations
* <strong>minimum support</strong> - (float: 0-1) threshold count of frequent FtGPs
* <strong>minimum representativity item</strong> - (float: 0-1) threshold count of transformations to be performed on the data-set

Example with a data-set and specified values<br>
```
TemporalGP -f datasets/DATASET.csv -t 0 -s 0.5 -r 0.5 -p 1
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
