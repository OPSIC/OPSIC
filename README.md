# OPSIC

This repository is for paper "Simultaneous Imputation and Clustering over Incomplete Data"

## File Structure

* code: source code of algorithms. For HC and kPOD, we use the open source implementations for them, i.e., HC (https://github.com/HoloClean/holoclean) and kPOD (https://github.com/iiradia/kPOD).
* data: dataset source files of all seven public data collections used in experiments.

## Dataset

- Iris: http://archive.ics.uci.edu/ml/datasets/Iris
- Glass: http://archive.ics.uci.edu/ml/datasets/Glass+Identification
- Wine: http://archive.ics.uci.edu/ml/datasets/Wine
- HTRU: https://archive.ics.uci.edu/ml/datasets/HTRU2
- Wholesale: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
- Hepatitis: https://sci2s.ugr.es/keel/dataset.php?cod=100
- Marketing: https://sci2s.ugr.es/keel/dataset.php?cod=163
- The real incomplete dataset ECG needs an internal assessment in the company and is not available for download thus far.

## Dependencies

* python 3.8
```
numpy==1.10.0
pandas==0.23.4
scipy==1.6.2
scikit-learn==0.24.1
gurobipy==9.5.0
```

## Instruction
```
python main.py
```
