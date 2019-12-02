# DS_hacks
Useful general functions for data science projects

## Getting Started
'DS_hacks' is a reposotory contains usefull functions I built (or copied from the wide internet) for when starting new DS projects.
I built this repo for my own needs, and the use of them is under your responsability. The functions in this repo was built while using google-colab, but should work for scripts as well.

## Dependencies:
* numpy
* pandas
* matplotlib
* seaborn
* missingno
* pandas-profiling
* pprint
* termcolor

# Documantation
## Table of Content
* Kaggle with notebooks
> * [load_kaggle_data](https://github.com/manelabinyamin/DS_hacks#load_kaggle_data)
> * [submit_prediction](https://github.com/manelabinyamin/DS_hacks#submit_prediction)
* Data cleaning
> * [nan_analysis](https://github.com/manelabinyamin/DS_hacks#nan_analysis)
> * [nan_cleaning](https://github.com/manelabinyamin/DS_hacks#nan_cleaning)
* Exploratory Data Analysis
> * [corrplot](https://github.com/manelabinyamin/DS_hacks#corrplot)
> * [plot_numeric_features](https://github.com/manelabinyamin/DS_hacks#plot_numeric_features)
> * [plot_line_numeric_over_numeric](https://github.com/manelabinyamin/DS_hacks#plot_line_numeric_over_numeric)
> * [plot_trend_numeric_over_numeric](https://github.com/manelabinyamin/DS_hacks#plot_trend_numeric_over_numeric)
> * [plot_numeric_over_categorical](https://github.com/manelabinyamin/DS_hacks#plot_numeric_over_categorical)
> * [plot_categorical_over_categorical](https://github.com/manelabinyamin/DS_hacks#plot_categorical_over_categorical)
* Feature engineering
> * TODO

## Functions

***

## Kaggle with notebooks
### load_kaggle_data()
**Description:** This function downloads the datasets straight to your colab notebook using kaggle's API.

**Arguments:**
* *Comp_api* (str) – The API of the data. To get it, copy the API from the competition website.

**How to use:**
Before using this function, you first need to have an API token. The API token can be downloaded from your Kaggle account. Save the file as `Kaggle.json` to your computer. The function may raise an error in the first run after resetting the runtime. If it happens, try to run it again.

**Output:**

The function will download the data to the notebook's memory. To read it use pandas' function `pd.read_csv(file_name.csv)`.
***
### submit_prediction()
**Description:** This function submits the prediction .cvs file to kaggle using kaggle's API.

**Arguments:**
* *comp_submission_api* (str) – The API of the submission. To get it, copy the API from the 'Submit Prediction' page.

**How to use:**
Before using this function, you first need to have an API token. The API token can be downloaded from your Kaggle account. Save the file as `Kaggle.json` to your computer. The argument `comp_submission_api` is a string contains the file name name and the submission's message.
The function may raise an error in the first run after resetting the runtime. If it happens, try to run it again.

**Output:**

The function will submit your prediction to kaggle's website.


***
## Data cleaning

### nan_analysis()

**Description:** 

This function plots the nans' patterns in the data.

**Arguments:**
* *df* (pd.DataFrame) – The DataFrame to analyze
* *figure_size* (list [width,height]) - the plots' size

**How to use:**

The output of this function will be readable for up to 50 features.

**Output:**

* bar-chart of the number of nans over features
* plot of nans over row. It also plots the number of not-nans over the rows.
* nans' correlation matrix

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/nan%20analysis/nan%20bar.PNG" alt="nans over features" width="300"/> <img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/nan%20analysis/nans%20in%20rows.PNG" alt="nans over rows" width="300"/> <img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/nan%20analysis/nan%20correlation.PNG" alt="nans correlation" width="300"/>

### nan_cleaning()

**Description:** 

This function removes rows and columns with too many nans, and then fills the remaining nans according to the desired method.
Be careful not to fill ID features, nor use them as predictors. You can specify ID features as 'str' to avoid filling and using them.

**Arguments:**
* *df* (pd.DataFrame) – The DataFrame to clean
* *row_nan_threshold* (float) - The ratio threshold for nans in each row. Rows with more nans will be removed. **Default = 0.3**.
* *col_nan_threshold* (float or dict) - The ratio threshold for nans in each columns. Columns with more nans than allowed will be removed. If dict was passed, the threshold for each column will be according to the specified value (1 for not specified features). **Default = 0.3**.
* *feature_type* (dict or None) - A dictionary that specifies the type of each feature (str/categorical/numeric). Every feature which was not sepecified in `feature_type` will be heuristically classified. **Default = None**
* *categorical_group_size* (int) - The average number of repetition of each value from which the feature will be classified as categorical. **Default = 10**.
* *print_feature_type* (bool) - whether or not to print the feature_type dict. **Default = True**.
* *method* (str) - What method to use for nan filling (median/best_predictor/minimum_impact). **Default = 'median'**.
* *alternative_method* (str) - What method to use if can't find valid predictor (median/minimum_impact). Only relevant when method='best_predictor'. **Default = 'minimum_impact'**.
* *num_of_bins* (int) - How many bins to use when method is 'minimum_impact'. **Default = 5**. 
* *inplace* (bool) - Whether or not to apply the changes on the given DataFrame. **Default = True**.

**How to use:**
* method `median` fills the feature with the median value (for numeric features) or most common value for categorical features.
* method `best_predictor` finds the feature (predictor), which best predicts the filled feature and then apply the `median` method after grouping the filled feature by the predictor.
* method `minimum_impact` minimizes the impact over the feature distribution.

The `nan_cleaning` function can't fill features of type 'str'.

**Output:**
The function returns the cleaned DataFrame. If 'inplace' is True, the function will update the original instance.

## Exploratory Data Analysis
### corrplot()

**Description:** 

This function plots a correlation matrix, which is more interpretable. 
The credits for this function belongs to [Drazen Zaric](https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)

**Arguments:**
* *df* (pd.DataFrame) – The DataFrame to plot

**How to use:**
* Send a DataFrame of numeric features

**Output:**
The function plots the correlation over the features.

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/corrplot.PNG" alt="correlation plot" width="500"/>


***
### plot_numeric_features()

**Description:** 
This function plots a grid of all numeric features with heatmaps over the features, scatter plot over the features and histograms.

**Arguments:**
* *df* (pd.DataFrame) – The DataFrame to plot
* *hue* (str) - Categorical feature to group all other features by. **Default = None**.
* *fig_size* (float) - The total size of the grid plot

**Output:**
The function plots a grid plot of all numeric features

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_numeric.PNG" alt="plot numeric" width="320"/> <img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_numeric_with_hue.PNG" alt="plot numeric with hue" width="380"/>

***
### plot_line_numeric_over_numeric()

**Description:** 
This function plots the line of two numeric features.

**Arguments:**
* *x* (str) - First numreric feature
* *y* (str) - Seconde numreric feature
* *df* (pd.DataFrame) – The DataFrame to plot
* *hue* (str) - Categorical feature to group the features by. **Default = None**.
* *fig_size* (list [width,height]) - The size of the plot.  **Default = [10,5]**.

**Output:**
This function plots line-plot of two numeric features

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_line_numeric_over_numeric.PNG" alt="plot line" width="300"/> <img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_line_numeric_over_numeric_with_hue.PNG" alt="plot line with hue" width="300"/>

***
### plot_trend_numeric_over_numeric()

**Description:** 
This function plots the trend between two numeric features. The function also estimates the regression line with the polynomial regression of the desired order.

**Arguments:**
* *x* (str) - First numreric feature
* *y* (str) - Seconde numreric feature
* *df* (pd.DataFrame) – The DataFrame to plot
* *hue* (str) - Categorical feature to group the features by. **Default = None**.
* *order* (int) - The polynom-order of the regression line. **Default = 1**.
* *fig_size* (list [width,height]) - The size of the plot.  **Default = [10,5]**.

**Output:**
This function plots a trend-plot of two numeric features. The regression is calculated for the desired order. 

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_trend_numeric_over_numeric.PNG" alt="trend line" width="300"/> <img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_trend_numeric_over_numeric_with_hue.PNG" alt="plot line with hue" width="300"/>

***
### plot_numeric_over_categorical()

**Description:** 
This function plots the relation between numeric and categorical features. The function plots the distribution of the numeric feature, hue by the categorical, the bar plot of the numeric values grouped by the categorical and a pie plot of the categorical feature.

**Arguments:**
* *num* (str) - The numreric feature
* *cat* (str) - The categorical feature
* *df* (pd.DataFrame) – The DataFrame to plot
* *fig_size* (list [width,height]) - The size of the plot.  **Default = [15,5]**.

**Output:**
Three plots describing the relation between the numeric and categorical features. 

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/plot_numeric_over_categorical.PNG" alt="numeric over categorical" width="750"/>

***
### plot_categorical_over_categorical()

**Description:** 
This function plots the correlation between two categorical features as the number of repetitions of all possible pairs of values.

**Arguments:**
* *cat1* (str) - The first categorical feature
* *cat2* (str) - The second categorical feature
* *df* (pd.DataFrame) – The DataFrame to plot

**Output:**
The correlation of two categorical features. 

<img src="https://github.com/manelabinyamin/DS_hacks/blob/master/images/EDA/categorical_over_categorical.PNG" alt="categoricals correlation" width="450"/>


## Authors

* [**Binyamin Manela**](https://github.com/manelabinyamin)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
