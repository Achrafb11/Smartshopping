# Smartshopping project repository for the IEEE Access paper.

###Important note : make sure to download the data.zip file available through this link :  
https://drive.google.com/file/d/1q-LkWMx5ar-OGlPPLFwSDi-IbLe7ZaIo/view?usp=sharing

###Extract the content of data.zip at the root of the project.

The 'ml_algorithms.py' script applies several machine learning algorithms to the given dataset and compares their performance using their metrics measurement.

Before running "ml_algorithms.py", you have to run two preprocessing scripts to ensure that data is formatted accordingly to the input expected by "ml_algorithms.py".

1. Go to "/Smartshopping_ieee/clustering" and run "clustering_steps.py" by typing on your terminal :
   

    cd clustering
   
    python clustering_steps.py


2. Go to "/Smartshopping_ieee/feature_engineering" and run "new_features.py" by typing on your terminal :
   

    cd ../feature_engineering
   
    python new_features.py


3. Finally, to run "ml_algorithms.py", open your terminal, go to "/Smartshopping_ieee/ml_algorithms":
   

    cd ../ml_algorithms
   
    python ml_algorithms.py

Note that the scripts were made using python 3.9.
**************************************************************
