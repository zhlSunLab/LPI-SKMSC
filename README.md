# LPI-SKMSC
A clustering-based lncRNA-protein interactions prediction method for handling imbalanced datasets.

## Dependency

python 3.6
pytorch 1.2.0

## Test

Visit the URL below to obtain result.zip and unzip it to the current directory：
https://drive.google.com/file/d/1u6q7tiyHYdXaPAOV4MWOSfhhQEwAp8S2/view?usp=sharing

Unzip the data.zip in the project to the current folder.

Test LPI-SKMSC：
```python
python val_test.py -f RPI1847
```

## Train

Unzip the data.zip in the project to the current folder.

Train LPI-SKMSC:
```python
python LPI_SKMSC.py -f RPI1847
```
