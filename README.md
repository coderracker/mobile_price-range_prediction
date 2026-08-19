In this project, I got a kaggle dataset from https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv and used it to train a model on 20 features to predict price range.

initially the data was already clean and everything was sorted.
So, I split the model using train test split, and tried finding the baseline model. There, I realised that logistic regression was performing well, over random forest, SVM and even KNeighbor classifier(performed poorly).
Using Logistic regression as my baseline model, I tried looking for better features. I checked correlation and found that Ram is highly correlated with the target feature.
I then checked with four highly correlated features, to see if they give a better prediction. That wasn't the case, when I tried with cross validation. 
I then checked if new features like screen ratio and total px(px_width * px_height) would work. they do not give better results.
Then I gave up on feature engineering, considering the corelation and multi-collinearity issues.
I then focused on fine-tuning Logistic regression, and since it was softmax regression being used by modern sklearn, I tried configuring C and solver for the model.
Using gridsearch, I found the optimal hyperparameters for Logistic regression.
I ran a CV using an optimally configured model, the CV score was 0.9735
I then saved the best model in /model using joblib
predict.py uses the stored model to predict.
requirements.txt are basics if have run ML projects before.

