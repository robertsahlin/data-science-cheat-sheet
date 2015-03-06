# data-science-cheat-sheet
Cheat sheet and best practice for data science

#Steps in a data science process
Define the business problem
Get the data
Explore the data
Build the model
Communicate and visualise
Build data product





#Feature Scaling and mean normalisation
*Why:* Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, the majority of classifiers calculate the distance between two points by the distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

*How:* Make sure features are on similar scale. Get every feature into approximately -1 =< x =< 1 range by dividing with range value of feature (i.e. max - min). Also subtract xi with the mean to make features have approximately zero mean (do not apply to x0 = 1). This method is widely used for normalization in many machine learning algorithms (e.g., support vector machines, logistic regression, and neural networks)[citation needed].

x1 <- (x1-mean(x1))/range(x1)

x1 <- (x1-mean(x1))/(max(x1) – min(x1))

I.e. the first feature (x1) of X1 has range(x1) = 2000 and mean(x1) = 1000, then, x1 = (x1-1000)/2000.
Example. x1 = 1500 -> x1 = (1500-1000)/2000 = 0,25

#Create features by combining features
Example area instead of frontage and depth.




#Don’t take default loss function for granted
Many practitioners train and pick the best model using the default loss function (e.g., squared error). In practice, off-the-shelf loss function rarely aligns with the business objective. Take fraud detection as an example. When trying to detect fraudulent transactions, the business objective is to minimize the fraud loss. The off-the-shelf loss function of binary classifiers weighs false positives and false negatives equally. To align with the business objective, the loss function should not only penalize false negatives more than false positives, but also penalize each false negative in proportion to the dollar amount. Also, data sets in fraud detection usually contain highly imbalanced labels. In these cases, bias the loss function in favor of the rare case (e.g., through up/down sampling).

Don’t use plain linear models for non-linear interaction

When building a binary classifier, many practitioners immediately jump to logistic regression because it’s simple. But, many also forget that logistic regression is a linear model and the non-linear interaction among predictors need to be encoded manually. Returning to fraud detection, high order interaction features like "billing address = shipping address and transaction amount < $50" are required for good model performance. So one should prefer non-linear models like SVM with kernel or tree based classifiers that bake in higher-order interaction features.

Don’t forget about outliers

Outliers are interesting. Depending on the context, they either deserve special attention or should be completely ignored. Take the example of revenue forecasting. If unusual spikes of revenue are observed, it's probably a good idea to pay extra attention to them and figure out what caused the spike. But if the outliers are due to mechanical error, measurement error or anything else that’s not generalizable, it’s a good idea to filter out these outliers before feeding the data to the modeling algorithm.

Some models are more sensitive to outliers than others. For instance, AdaBoost might treat those outliers as "hard" cases and put tremendous weights on outliers while decision tree might simply count each outlier as one false classification. If the data set contains a fair amount of outliers, it's important to either use modeling algorithm robust against outliers or filter the outliers out.

Don’t use high variance model when n<<p

SVM is one of the most popular off-the-shelf modeling algorithms and one of its most powerful features is the ability to fit the model with different kernels. SVM kernels can be thought of as a way to automatically combine existing features to form a richer feature space. Since this power feature comes almost for free, most practitioners by default use kernel when training a SVM model. However, when the data has n<<p (number of samples << number of features) --  common in industries like medical data -- the richer feature space implies a much higher risk to overfit the data. In fact, high variance models should be avoided entirely when n<<p.

Don’t do L1/L2/... regularization without standardization

Applying L1 or L2 to penalize large coefficients is a common way to regularize linear or logistic regression. However, many practitioners are not aware of the importance of standardizing features before applying those regularization.

Returning to fraud detection, imagine a linear regression model with a transaction amount feature. Without regularization, if the unit of transaction amount is in dollars, the fitted coefficient is going to be around 100 times larger than the fitted coefficient if the unit were in cents. With regularization, as the L1 / L2 penalize larger coefficient more, the transaction amount will get penalized more if the unit is in dollars. Hence, the regularization is biased and tend to penalize features in smaller scales. To mitigate the problem, standardize all the features and put them on equal footing as a preprocessing step.

Regularization methods are used for model selection, in particular to prevent overfitting by penalizing models with extreme parameter values. The most common variants in machine learning are L1 and L2 regularization, which can be added to learning algorithms that minimize a loss function E(X, Y) by instead minimizing E(X, Y) + a|w|, where w is the model's weight vector, |•| is either the L1 norm or the squared L2 norm, and a is a free parameter that needs to be tuned empirically. This method applies to many models. When applied in linear regression, the resulting models are termed ridge regression or lasso, but regularization is also employed in (binary and multiclass) logistic regression, neural nets, support vector machines, conditional random fields and some matrix decomposition methods. L2 regularization may also be called "weight decay", in particular in the setting of neural nets.

L1 regularization is often preferred because it produces sparse models and thus performs feature selection within the learning algorithm, but since the L1 norm is not differentiable, it may require changes to learning algorithms, in particular gradient-based learners.

Don’t use linear model without considering multi-collinear predictors

Imagine building a linear model with two variables X1 and X2 and suppose the ground truth model is Y=X1+X2. Ideally, if the data is observed with small amount of noise, the linear regression solution would recover the ground truth. However, if X1 and X2 are collinear, to most of the optimization algorithms' concerns, Y=2*X1, Y=3*X1-X2 or Y=100*X1-99*X2 are all as good. The problem might not be detrimental as it doesn't bias the estimation. However, it does make the problem ill-conditioned and make the coefficient weight uninterpretable.

Interpreting absolute value of coefficients from linear or logistic regression as feature importance

Because many off-the-shelf linear regressor returns p-value for each coefficient, many practitioners believe that for linear models, the bigger the absolute value of the coefficient, the more important the corresponding feature is. This is rarely true as (a) changing the scale of the variable changes the absolute value of the coefficient (b) if features are multi-collinear, coefficients can shift from one feature to others. Also, the more features the data set has, the more likely the features are multi-collinear and the less reliable to interpret the feature importance by coefficients.

So there you go: 7 common mistakes when doing ML in practice. This list is not meant to be exhaustive but merely to provoke the reader to consider modeling assumptions that may not be applicable to the data at hand. To achieve the best model performance, it is important to pick the modeling algorithm that makes the most fitting assumptions -- not just the one you’re most familiar with.

Building a model
The place to start is exploratory data analysis (EDA). This entails making plots and building intuition for your particular dataset. EDA helps out a lot, as well as trial and error and iteration. To be honest, until you’ve done it a lot, it seems very mysterious. The best thing to do is start simply and then build in complexity. Do the dumbest thing you can think of first. It’s probably not that dumb. For example, you can (and should) plot histograms and look at scatterplots to start getting a feel for the data. Then you just try writing something down, even if it’s wrong first (it will probably be wrong first, but that doesn’t matter). So try writing down a linear function (more on that in the next chapter). When you write it down, you force yourself to think: does this make any sense? If not, why? What would make more sense? You start simply and keep building it up in complexity, making assumptions, and writing your assumptions down. You can use full-blown sentences if it helps—e.g., “I assume that my users naturally cluster into about five groups because when I hear the sales rep talk about them, she has about five different types of people she talks about”—then taking your words and trying to express them as equations and code. Remember, it’s always good to start simply. There is a trade-off in modeling between simple and accurate. Simple models may be easier to interpret and understand. Oftentimes the crude, simple model gets you 90% of the way there and only takes a few hours to build and fit, whereas getting a more complex model might take months and only get you to 92%. You’ll start building up your arsenal of potential models throughout this book. Some of the building blocks of these models are probability distributions.

Exploratory data analysis
Plotting data and making comparisons can get you extremely far, and is far better to do than getting a dataset and immediately running a regression just because you know how. It’s been a disservice to analysts and data scientists that EDA has not been enforced as a critical part of the process of working with data. Take this opportunity to make it part of your process.

There are important reasons anyone working with data should do EDA. Namely, to gain intuition about the data; to make comparisons between distributions; for sanity checking (making sure the data is on the scale you expect, in the format you thought it should be); to find out where data is missing or if there are outliers; and to summarize the data.

In EDA, there is no hypothesis and there is no model. The “exploratory” aspect means that your understanding of the problem you are solving, or might solve, is changing as you go.

The basic tools of EDA are plots, graphs and summary statistics. Generally speaking, it’s a method of systematically going through the data, plotting distributions of all variables (using box plots), plotting time series of data, transforming variables, looking at all pairwise relationships between variables using scatterplot matrices, and generating summary statistics for all of them. At the very least that would mean computing their mean, minimum, maximum, the upper and lower quartiles, and identifying outliers.

EDA also helps with debugging the data collection process.

In the course of doing EDA, we may realize that it isn’t actually clean because of duplicates, missing values, absurd outliers, and data that wasn’t actually logged or incorrectly logged. If that’s the case, we may have to go back to collect more data, or spend more time cleaning the dataset.
