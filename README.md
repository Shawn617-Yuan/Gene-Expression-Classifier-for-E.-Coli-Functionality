# Gene Expression Classifier for E. Coli Functionality
**This project focuses on developing a classifier to identify the functionality of genes in E. coli. Utilizing a dataset with 1500 data points, the project involved implementing various preprocessing techniques and classification models to accurately predict gene functionality related to cell communication in E. coli.**

## Data Description
- **Dataset:** 'Ecoli.csv' from a micro-array expression dataset.  
- **Features:** 103 numerical and 3 nominal features describing gene function, with the final column as the target label.  
- **Target:** Indicating the functionality of the gene in cell communication.

## Preprocessing and Analysis
- **Outlier Detection:** Density-based, Model-based, and Isolation-based techniques.  
- **Imputation Methods:** Class-specific value imputation.  
- **Normalization:** Max-min normalization and Standardization (z-score normalization).  
- **Cross-Validation:** Used to determine the effectiveness of preprocessing techniques.

## Classification Models and Tuning
- **Models Used:** Decision Tree, Random Forest, K-Nearest Neighbour, and Naïve Bayes.  
- **Hyperparameter Tuning:** Employed GridSearchCV for optimal model tuning.  
- **Best Model:** RandomForestClassifier, outperforming others in terms of accuracy and F1-score.

## Results
- **Project Score:** Achieved 19.5 out of 20 points, ranking in the top 5% among peers.  
- **Accuracy:** Approximately 82%, with an 8% improvement over baseline models.  
- **Key Finding:** The combination of class-specific imputation, model-based outlier detection, and max-min normalization was most effective in enhancing model performance.

## Future Work
Further exploration of advanced models and deep learning techniques could be conducted to improve classification accuracy and handle larger, more complex datasets.
