## Wine Classifier 

### About
* We were tasked with classifying a set of wine samples into a certain number of classes. We then extract discriminative features from the data sample and then build a classifier in [wine_classifier.py](wine_classifier.py) that uses such features to recognise what class a given sample belongs to.
* We made use of a dataset which consists of 178 samples each corresponding to a wine.
* Each sample contains 13 features/dimensions, corresponding to a chemi-
cal constituent of the wine: 1) Alcohol, 2) Malic acid, 3) Ash, 4) Alkalinity of
ash, 5) Magnesium, 6) Total phenols, 7) Flavanoidsm, 8) Nonflavanoid phe-
nols, 9) Proanthocyanins, 10) Color intensity, 11) Hue, 12) OD280/OD315
of diluted wines, 13) Proline.
* We want to be able to tell what cultivar each wine derives from. 
* We then proceded to implement ```K Nearest Neighbours```, ```Naive Bayes``` and ```Principal Component Analysis(PCA)```.
* We then conducted an in-depth study of how we went this process and analysis of the results in [SPS_REPORT.pdf](SPS_REPORT.pdf).

### How to Run 
* Run the command ```python wine_classifier.py``` which should then call each of the functions above respectively and produce the same results detailed in [SPS_REPORT.pdf](SPS_REPORT.pdf).


