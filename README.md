# MIP-SVM-QCQP
MIP for modelling fairness constraints

1. The paper link is: ...
2. Datasets used whilst running constructed algorithm: 
            Adult(d=121?): _Predicts whether income exceeds $50K/yr based on census data._
            German(d=59?): _Classifies people described by a set of attributes as good or bad credit risks._
            COMPAS_ext(d=406?): _Used to predict recidivism (whether a criminal will reoffend or not) in the USA._
            ------------------
            The format of these datasets is entirely numerical. 
            
3. How to run the code:
          make clean
          make all
          make execute
          
4. Calculate metrics: python ./metrics_svm_simple.py metrics.txt
