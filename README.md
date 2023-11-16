# LPI-SKMSC
A clustering-based lncRNA-protein interactions prediction method for handling imbalanced datasets.

The current code is the test part of LPI-SKMSC. Predictive metrics can be obtained by testing the trained model. We will submit the complete code soon.

## Dependency

python 3.6
pytorch 1.2.0

## Test

Visit the URL below to obtain result.zip and unzip it to the current directory：
https://drive.google.com/file/d/1u6q7tiyHYdXaPAOV4MWOSfhhQEwAp8S2/view?usp=sharing

Unzip the data.zip in the project to the current folder.

Test LPI-SKMISC under different datasets：
```python
python val_test.py -f RPI1847
python val_test.py -f RPI7317
python val_test.py -f RPI488
```
