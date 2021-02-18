# Machine Learning Training

The objective of this training is to allow you to grasp the essential concepts that help you get started with Machine Learning.
This training covers the vast majority of ideas and concepts that you should encounter when solving a machine learning problem.
The main framework used in this training is scikit-learn.

Please follow the order of folders and files the way they are sorted and numbered.

The package versions used in this training have been freezed to requirements.txt

Here is the outline of the training:


* Prerequisites: 
    * Linear Algebra Review 
        * Vectors, Matrices and Tuples
        * Addition, Multiplication, Inverse, Transpose…
    * Multivariate Calculus (coming soon)
        * Functions
        * Differential Calculus / partial derivatives
    
* Introduction to Machine Learning 
    * What is Machine Learning
    * Applications
    * Supervised vs Unsupervised
    * Introducing ML with Univariate Linear Regression
        * Model Representation
        * Cost Function
        * Optimization
            * Ordinary Least Squares
                * Non-invertibility
                * Computationally expansive
            * Gradient Descent
            * Batch, Mini Batch & Stochastic Gradient Descent
                
* Practical Aspects of Machine Learning
    * Data Splitting
        * Validation
        * Cross-Validation
    * Bias / Variance Diagnosis 
    * Bias / Variance Correction 
    * Regularization 
    * Learning Curves 
        * Learning Rate
        * Learning Rate Decay
    * Cost Functions & Optimization Techniques 
    * Regression and Classification Evaluation Metrics 
        * MSE, RMSE, MAE, RSquared ...
        * Confusion Matrix, Accuracy, Precision, Recall, F1score ...
        * ROC, AUC
        * Error metrics for skewed classes
    * Hyperparameters tuning
        * Grid Search
        * Random Search
    * Feature Representation & Engineering
        * Data Leakage
        * Missing Values Imputation
        * Numerical Features
            * Normalization
            * Log Scaling
            * Bucketing
        * Categorical Features
            * Label encoding 
            * One hot encoding
        * Outliers
        * Images (coming soon)
        * Text (coming soon)
        
* Regression
    * Multiple Linear Regression
        * Multiple Features
        * Features and Polynomial regression
        * Gradient descent for multiple variables
    * Polynomial Regression
    * Lasso Regression (L1 regularization)
    * Ridge Regression (L2 regularization)
    * Elastic Net Regression 
    * Decision Trees Regression
    
* Classification
    * Logistic Regression
        * Decision Boundary
        * Cost Function
    * Multiclass Classification
        * One-vs-all
        * Multinomial classification
    * Support Vector Machines (coming soon)
    * Decision Trees
    * Ensemble  Methods (Bagging, Boosting, Stacking). 
        * Bagged Trees
            * Random Forest
        * Boosted Trees 
            * Gradient Boosting

* Unsupervised Learning
    * Clustering
        * Partition methods
            * k-means clustering
            * Cluster validity, choosing the number of clusters
                * Elbow method
                * Average silhouette method
                * Gap statistic method
        * Hierarchical methods
        * Distribution models (coming soon)
            * Gaussian Mixture Models & EM
        * Density Models (coming soon)
            * DBSCAN
    * Dimensionality Reduction (coming soon)
        * Principal Component Analysis