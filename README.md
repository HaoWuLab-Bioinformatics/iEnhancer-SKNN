# iEnhancer-SKNN
iEnhancer-SKNN is a stacking ensemble learning model for accurately predicting enhancers and their strength.

## Dataset
The file "dataset" is the DNA sequence dataset used in this study. 
The training set contained 742 strong enhancer sequences, 742 weak enhancer sequences and 1484 non-enhancer sequences.
The independent test set contains 100 strong enhancer sequences, 100 weak enhancer sequences, and 200 non-enhancer sequences.
In enhancer recognition, strong enhancers and weak enhancers are taken as positive examples, while non-enhancers are taken as negative examples. 
In enhancer strength classification, strong enhancers are taken as positive examples and weak enhancers as negative examples.

## Overview
Features Kmer, PseDNC and PCPseDNC are extracted using repDNA, and Z-Curve9 is extracted using iLearnPlus.
The code "model.py" is used for model training and performance evaluation. We perform 10-fold cross-validation on the training set and evaluate the performance of iEnhancer-SKNN on the independent test set.
The file "Example" is the feature KMER used in this study, including the kmer-based features of the training set and independent test set and their corresponding labels respectively.

## Dependency
Python 3.6
sklearn 
numpy 
mlxtend 

## Usage
First, you should download repDNA (http://bioinformatics.hitsz.edu.cn/repDNA/) to extract features Kmer (K =1, 2, 3), PseDNC (Lamada =3, W =0.05) and PCPseDNC (Lamada =3, W =0.05).
Then, download iLearnPlus (https://github.com/Superzchen/iLearnPlus/) to extract the feature z-Curve9.
Finally, if you want to compile and run iEnhancer-SKNN, you can run the script as:
`python model.py`

