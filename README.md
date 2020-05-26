# Sentiment Analysis on Yelp Data
Sentiment analysis plays a crucial role in social media monitoring, particularly, for users’ 
opinion mining. In our research project, we aim to analyze the user reviews for restaurants from 
the Yelp dataset. We try to quantify the most impactful positive and negative sentiments expressed 
by a user for a restaurant belonging to a specific category. As a baseline model, we first predict 
the user ratings using the sentiments in the user reviews. For this, we make use of algorithms 
such as Multinomial Naïve Bayes, Random Forest Classifier and Multilayer Perceptron. We then 
record the accuracy of predicted ratings versus the actual ratings using the precision. 
We get a score of 95.37 for the Multilayer Perceptron, 93.02 for the Multinomial Naïve Bayes.

Our aim is then to filter out the less impactful positive and negative words based on a 
threshold value of the polarity of the occurring sentiments. To calculate the polarity, 
we first fit a Linear Support vector classifier to classify the review text into positive 
and negative words based on a bag of words. Using the coefficient of the support vector 
classifier, the frequency of the words and the total number of words, we calculate the 
polarity. After this, we again predict the user ratings given the user reviews, but this 
time, we consider only those reviews that contain the impactful positive words. The ratings 
predicted from the meaningful sentiments are then compared with the baseline model for 
quantification. We find out that the score of the Multinomial Naive Bayes model is 70.42. 
The second aim of this project is to recommend those restaurants to users that have been 
positively reviewed with the most impactful sentiments. The results of this project will 
show the difference between positive and meaningfully positive reviews. The restaurants will 
be predicted based on this ‘helpful’ sentiment. The impactful negative reviews can also be 
used in the future to help the restaurant owner improve their business.

### Intoduction
Research Idea : Quantifying the most impactful words that can determine the sentiment of a text
Sentiment Analysis uses NLP, statistics, or machine learning methods to extract, identify, 
or otherwise characterize the sentiment content of a text unit. It is sometimes refered to 
as opinion mining, although the emphasis in this case is on extraction.

Sentiment Analysis has been tackled mainly from two different approaches : 
computational learning techniques and semantic approaches. 
i. Semantic approaches: These are characterized by the use of dictionaries of words 
(lexicons) with semantic orientation of polarity or opinion. Systems typically preprocess 
the text and divide it into words, with proper removal of stop words and a linguistic 
normalization with stemming or lemmatization, and then check the presence or absence of 
each term of the lexicon, using the sum of the polarity values of the terms for assigning 
the global polarity value of the text.

ii. Learning-based : It focuses on training a classifier using any supervised learning 
algorithm from a collection of annotated texts, where each text is usually represented by 
a vector of words (bag of words), n-grams or skip-grams, in combination with other types 
of semantic features that attempt to model the syntactic structure of sentences, intensification,
 negation, subjectivity or irony.

In traditional approaches on sentiment analysis, the sentiments are just classified 
according to mood and results are predicted accordingly. We are however, taking a different 
approach using sentiment analysis to understand the impact of certain positive words on the 
ratings.

### The Dataset : Yelp dataset taken from Kaggle
Kaggle link: https://www.kaggle.com/yelp-dataset/yelp-dataset

### Data Description
In total, there are 5,200,000 user reviews, information on 174,000 business. 
We will focus on two tables which are business table and review table.

### Attributes of business table are as following:

business_id: ID of the business
name: name of the business
neighborhood
address: address of the business
city: city of the business
state: state of the business
postal_code: postal code of the business
latitude: latitude of the business
longitude: longitude of the business
stars: average rating of the business
review_count: number of reviews received
is_open: 1 if the business is open, 0 therwise
categories: multiple categories of the business
Attribues of review table are as following:

review_id: ID of the review
user_id: ID of the user
business_id: ID of the business
stars: ratings of the business
date: review date
text: review from the user
useful: number of users who vote a review as usefull
funny: number of users who vote a review as funny
cool: number of users who vote a review as cool
Evaluation Metric
Confusion Matrix: Confusion Matrix as the name suggests gives us a matrix as output and 
describes the complete performance of the model.
Lets assume we have a binary classification problem. We have some samples belonging to two 
classes : YES or NO. Also, we have our own classifier which predicts a class for a given 
input sample. On testing our model on 165 samples ,we get the following result.

### Confusion Matrix
There are 4 important terms :

True Positives(TP) : The cases in which we predicted YES and the actual output was also YES.
True Negatives(TN) : The cases in which we predicted NO and the actual output was NO.
False Positives(FP) : The cases in which we predicted YES and the actual output was NO.
False Negatives(FN) : The cases in which we predicted NO and the actual output was YES.

Accuracy for the matrix can be calculated by taking average of the values lying across 
the “main diagonal”

i.e (TP+FN)/No. of Samples
Confusion Matrix forms the basis for the other types of metrics.

F1 Score: F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score 
is [0, 1]. It tells you how precise your classifier is (how many instances it classifies 
correctly), as well as how robust it is (it does not miss a significant number of instances).
High precision but lower recall, gives you an extremely accurate, but it then misses a large 
number of instances that are difficult to classify. The greater the F1 Score, the better is 
the performance of our model. It can be mathematically calculated as: 
F1 = 2*(1/(1/precision)+(1/recall))

F1 Score tries to find the balance between precision and recall.

Precision : It is the number of correct positive results divided by the number of positive 
results predicted by the classifier. i.e TP/(TP+FP)

Recall : It is the number of correct positive results divided by the number of all 
relevant samples (all samples that should have been identified as positive) i.e TP/(TP+FN)

image.png

### Citations
https://en.wikipedia.org/wiki/Multilayer_perceptron
https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/
https://www.kaggle.com/omkarsabnis/sentiment-analysis-on-the-yelp-reviews-dataset
https://www.kaggle.com/yelp-dataset/yelp-dataset/kernels
https://github.com/zoehuang7/Yelp-Review-Analysis/blob/master/Group%2021%20-%20Yelp%20Reviw%20Analysis.ipynb
https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/

### Conclusion
We have executed 3 models above to predict the user ratings based on the user reviews.
We observe that Multinomial Naive Bayes gives the best score of 70.42.
This may not be better than the baseline model, but that can be explained by the fact that 
we have considered lesser data than that of the baseline model.
Since we are taking only the positive reviews and polarity above, the precision for the 5 star 
ratings is pretty good, that is, 0.71
Thus we can conclude that just a subset of all the review text is the most determinant for 
the user ratings, although not as good as the whole set.

This been said, we shouldn't completely ignore the words that have less impact on the review.

On the whole it is better to keep the threshold value as low as possible so that it can 
consider more impactful words.

Thus we have justified the importance of impactfulness of words in Sentiment Analysis.

image.png

### License
Copyright 2019 TANUSHREE DESHMUKH, PRITHVIRAJ PATIL, VARADA DHARMADHIKARI

Permission is hereby granted, free of charge, to any person obtaining a copy of this 
software and associated documentation files (the "Software"), to deal in the Software 
without restriction, including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons 
to whom the Software is furnished to do so, subject to the following conditions: The above 
copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.