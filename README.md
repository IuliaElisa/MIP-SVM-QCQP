# MIP-SVM-QCQP
MIP for modelling fairness constraints

1. The paper link is: ...
2. Datasets used whilst running constructed algorithm: 
            Adult(d=121?): _Predicts whether income exceeds $50K/yr based on census data._
            German(d=59?): _Classifies people described by a set of attributes as good or bad credit risks._
            COMPAS_ext(d=406?): _Used to predict recidivism (whether a criminal will reoffend or not) in the USA._
            
The format of these datasets is entirely numerical. 
            
3. How to run the code:
          make clean
          make all
          make execute
          
4. Calculate metrics: python ./metrics_svm_simple.py metrics.txt
5. For files run on Calcul server, the total time for code execution for a specific combination of parameters can be found in _some_name-nb_job.stderr_ where _nb_job_ is the id of the submitted job. The output of the code from screen (if any) is redirected towards the file _some_name-nb_job.stdout._
