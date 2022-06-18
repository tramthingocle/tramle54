############# BSAN 460 -- Business Analytics Senior Project ##############
################################ Group 2 #################################
### Andrea Ngo | Vinh Nguyen | Tram Le | Paige Powell | Nhat Ha Nguyen ###

################# STOCK MARKET TWEETS - SENTIMENT ANALYSIS ###############

######################### Prof. Irina Nedelcu ############################
############################## 11/29/2021 ################################

## Business questions/problem:
## 1. Build a model that accurately predicts stock sentiment: How public tweets opinions affect stock performance?
## 2. Stock forecast/prediction if machine learning performance is strong


rm(list=ls())
# set the working directory and read files
getwd() 
setwd("/Users/haianh.ngo1/Downloads")
setwd("D:/Documents/DREXEL SENIOR/FALL2021/BSAN460")

# Load/ install packages that we will need in the analysis:
install.packages("tm")
install.packages("textstem")
install.packages("tidytext")
install.packages("wordcloud")
install.packages("textdata")
install.packages("stringr")   
install.packages(c("randomForest", 
                   "ipred"))
install.packages('e1071')
library(stringr)
library(readr)
library(dplyr)
library(tm)
library(textstem)
library(tidytext) 
library(textdata)
library(wordcloud)
library(DescTools)
library(caret)
library(syuzhet)
library(e1071)
library(randomForest)
library(pROC)

# A. DATA OVERVIEW

# Import csv file and view intial structure
tweets <- read_csv("Tweets stock market.csv")

# Out of 5000 rows, only 1300 were manually labelled.

# We would subset the original csv file to only index those rows that were labelled manually.
tweets_labelled <- tweets[complete.cases(tweets),]
View(tweets_labelled)
head(tweets_labelled)

# check the structure of the data
str(tweets_labelled)
# summary
summary(tweets_labelled)

#Data Exploration
Abstract(tweets_labelled)

# Transform target variable sentiment to factor
tweets_labelled$sentiment <- factor(tweets_labelled$sentiment)

#Plotting the distribution of target variable
plot(tweets_labelled$sentiment,
     main = "Stock Sentiment",
     xlab = "Sentiment")
#Out of 1300 labelled tweets, we have around more than 500 positive sentiments, about 350 negative, and > 400 neutral. 

# Checking the column names
colnames(tweets_labelled)


# B. TEXT PREPROCESSING AND TF-IDF


#B.1: Text Preprocessing

#Remove Special Characters
tweets_labelled <- tweets_labelled %>% mutate(new_text_noSpecial = gsub("[^0-9A-Za-z///' ]", "" ,text ,ignore.case = TRUE))
#This will remove hashtags too.

# Transform to all lower cases
tweets_labelled <- tweets_labelled %>% mutate(new_text_lower = tolower(new_text_noSpecial))

# Remove numbers
tweets_labelled <- tweets_labelled %>% mutate(new_text_noNumbers = gsub('[[:digit:]]','',new_text_lower)) 

# Remove the stopwords:
stopwords_regex = paste(c("...",stopwords('en')), collapse = '\\b|\\b')
tweets_labelled <- tweets_labelled %>% mutate(new_text_noStopWords = gsub(stopwords_regex,'',new_text_noNumbers)) 

# Remove @username #
tweets_labelled <- tweets_labelled %>% mutate(new_text_noTags = gsub('@\\w*','',new_text_noStopWords))

# Remove URLs from text #
tweets_labelled <- tweets_labelled %>% mutate(new_text_noURL = gsub('http.*\\s','',new_text_noTags))


# Replace words within text #
# If there are words that have typos, let's change them to the correct words.
tweets_labelled <- tweets_labelled %>% mutate(new_text_noTypos = gsub('fb','facebook',new_text_noURL))

# Remove extra white space #
# (this would include space, tab, vertical tab, newline, form feed, carriage return):
tweets_labelled <- tweets_labelled %>% mutate(new_text_noSpaces = gsub('\\s+',' ',new_text_noTypos))

# Apply lemmatization #
tweets_labelled <- tweets_labelled %>% mutate(new_text_Lemma = lemmatize_strings(new_text_noSpaces))

# Apply stemming #
tweets_labelled <- tweets_labelled %>% mutate(new_text_Stem = stem_strings(new_text_noSpaces))
# Keep just the Text and new_text_Lemma columns for further modeling:
#tweets_labelled <- tweets_labelled %>% select(text, new_text_Lemma)


# Transform the tweets_labelled data frame to corpus. 
# We need to transform it to corpus because the DocumentTermMatrix() function takes a corpus as an input.
corp <- Corpus(VectorSource(tweets_labelled$new_text_Lemma))
class(corp)
length(corp)
inspect(corp[1])


# B.2 Tf-Idf
# Once you create Tf-Idf, you can use it as input to predictive models
##########
# create the Tf-Idf
my_tfidf <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf))
my_tfidf # documents: 5000, terms: 15345
#view high-level information about our document term matrix
inspect(my_tfidf)

## Dimension Reduction

#Minimum Document Frequency helps us specify the desired minimum number of documents a term must appear in. Let's set the minimum frequency to 5.
my_tfidf5 <- DocumentTermMatrix(x = corp, 
                                control = list(bounds = list(global = c(5, Inf))))
my_tfidf5

# Let's take a look at the terms in our my_tfidf5
Terms(my_tfidf5)

# If data is bigger, Tf-Idf can get quite large, since it has a column for each term
# Tf-Idf has many terms, so lets, remove those that are sparse, those that appear only a few times
# 0.99 gives you 180 terms. change the number to lower or higher to get less or more terms 
# # 0.99 tells that a column must have at most 99% of the columns with 0 in them
my_tfidf_small <-  removeSparseTerms(my_tfidf5, 0.99) 
my_tfidf_small  # documents: 1300, terms: 67

# Let's find those terms that occurs at least 25 times and 50 times in our corpus
findFreqTerms(x = my_tfidf_small, lowfreq = 25)
findFreqTerms(x = my_tfidf_small, lowfreq = 50)


# transform the my_tfidf_small into a data frame
stock_data_frame <- as.data.frame(as.matrix(my_tfidf_small))
stock_data_frame <- cbind(sentiment_dependent = tweets_labelled$sentiment, stock_data_frame)

# Now, that the data is stored as a tf-idf, we can use it in predictive modeling.
# The Sentiment column would be the dependent variable, while the other columns are the independent variables.


# C. VISUALIZATION

# Word Cloud to visualize our corpus
set.seed(2)
wordcloud(corp, # corpus object
          random.order = FALSE, # most frequent in center
          colors = brewer.pal(8, "Dark2"), # color schema
          max.words = 150) # top 150 terms

# Extract top 25 stock tickers
ticker_pattern <- str_extract(tweets_labelled$text, "[$][A-Z]+")
top_ticker <- sort(table(x = ticker_pattern), decreasing = TRUE)
top25_ticker <- head(top_ticker, n = 25)

top25_ticker <- as.data.frame(top25_ticker)

# Plot top 25 stocks
library(ggplot2)
ggplot(data = top25_ticker, aes(x = x, y = Freq)) + 
  geom_bar(stat = "identity", fill = "#30b3a2") +
  coord_flip( ) +
  labs(title = "Top 25 Stock Tickers", x = "Ticker", y = "Frequency") +
  geom_text(
    aes(label = Freq, y = Freq + 25),
    position = position_dodge(0.75),
    vjust = 0.25)


# D. TEXT MINING MODELS

# D.1: Machine Learning Classification Method - Random Forest

# D.1.1: Base Model

# Reducing our dependent Variable sentiment to 2 classes (positive, negative) for better performance.
# We transform the sentiment column so that it only has 2 factors: positive or negative. 
# All the neutral sentiments will be omitted in the training set.
#First, let's create a copy of our tf-idf
bi_stockDf <- stock_data_frame

#Subsetting only the rows with positive and negative sentiment
data_2class <- bi_stockDf[(bi_stockDf == "positive" | bi_stockDf == "negative"),] 
#Removing empty rows
data_2class_full<- data_2class[complete.cases(data_2class), ]

# Letting R knows that we now only have 2 classes.
data_2class_full$sentiment_dependent <- as.character(data_2class_full$sentiment_dependent)
data_2class_full$sentiment_dependent <- as.factor(data_2class_full$sentiment_dependent)





# Let's split our data into training & testing set with 85/15 split.
set.seed(2)
sub <- createDataPartition(y = data_2class_full$sentiment_dependent, # target variable
                           p = 0.85,#85% will be used for training, the remaining 15% for testing
                           list = FALSE)
# Let's subset the rows of stock_data_frame to create the training dataframe, those not in sub will belong to the testing df.
bi_train <- data_2class_full[sub, ] 
bi_test <- data_2class_full[-sub, ]

Desc(bi_train$sentiment_dependent)
# The frequency of positive (60.3%) is slight higher than negative(39.7%). Class Imbalance doesn't seem to be an issue here.


##################################################
################### Analysis ####################
##################################################

# 1. Missing values
# Visualize using the PlotMiss() function 
# in the DescStats package
PlotMiss(x = data_2class_full, 
         main = "Missing Values by Variable")
# There should be no NAs after our transformation.

# 3. Analysis
# initialize random seed
set.seed(2) 

rf_mod <- randomForest(formula = sentiment_dependent ~. , # use all other variables to predict sentiment
                       data = bi_train%>%select(-`break`), # training data
                       importance = TRUE, # obtain variable importance 
                       ntree = 500) # number of trees in forest

# We can view basic output from the model
rf_mod


# Variable Importance Plot
# We can view the most important variables 
# in the Random Forest model using the
# varImpPlot() function
varImpPlot(x = rf_mod, # randomForest object
           main = "Variable Importance Plot") # title

## Training Performance
# We use the predict() function to generate 
# class predictions for our training set
base.RFpreds <- predict(object = rf_mod, # RF model
                        type = "class") # class predictions

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# training dataset (train).
RF_btrain_conf <- confusionMatrix(data = base.RFpreds, # predictions
                                  reference = bi_train$sentiment_dependent, # actual
                                  positive = "positive",
                                  mode = "everything")
RF_btrain_conf

## Testing Performance
# We use the predict() function to generate 
# class predictions for our testing set
base.teRFpreds <- predict(object = rf_mod, # RF model
                          newdata = bi_test, # testing data
                          type = "class")

#Convert to factors before comparison
stock_data_frame$sentiment_dependent <- as.factor(stock_data_frame$sentiment_dependent)


# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# testing dataset (test).
RF_btest_conf <- confusionMatrix(data = base.teRFpreds, # predictions
                                 reference = bi_test$sentiment_dependent, # actual
                                 positive = "positive",
                                 mode = "everything")
RF_btest_conf


## doing better job at predicting negative class rather than positive,
## but overall better model than bagging?

## Goodness of Fit

# To assess if the model is balanced,
# underfitting or overfitting, we compare
# the performance on the training and
# testing. We can use the cbind() function
# to compare side-by-side.

# Overall
cbind(Training = RF_btrain_conf$overall,
      Testing = RF_btest_conf$overall)

# Class-Level
cbind(Training = RF_btrain_conf$byClass,
      Testing = RF_btest_conf$byClass
      )


options(scipen = 999)



# D.1.2: Tuned Model

### Hyperparameter Tuning

# We will tune the number of variables to 
# randomly sample as potential variables to split on 
# (m, the mtry argument).

# We use the tuneRF() function in the 
# randomForest package. The output will
# be a plot, where we choose the mtry with
# the smallest OOB Error. By setting
# doBest = TRUE, the best mtry will
# be used to automatically create
# an RF model

set.seed(2) # initialize random seed

tuneR <- tuneRF(x = bi_train[, -1], # use all variable except the 1st (sentiment_dependent) as predictors
                y = bi_train$sentiment_dependent, # use sentiment_dependent as the target variable
                ntreeTry = 500, # 500 trees in the forest
                doBest = TRUE) # use the best m (mtry) value to create an RF model
## most minimal mtry is 8, with the lowest OOB error

# View basic model information
tuneR

# View variable importance for the tuned 
# model
varImpPlot(tuneR)
## sto and trad are the 2 most important variables

## Training Performance
# We use the predict() function to generate 
# class predictions for our training set
tune.trRFpreds <- predict(object = tuneR, # tuned RF model
                          type = "class") # class predictions

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# training dataset (train).
RF_ttrain_conf <- confusionMatrix(data = tune.trRFpreds, # predictions
                                  reference = bi_train$sentiment_dependent, # actual
                                  positive = "positive",
                                  mode = "everything")
RF_ttrain_conf


## Testing Performance
# We use the predict() function to generate 
# class predictions for our testing set
tune.teRFpreds <- predict(object = tuneR, # tuned RF model
                          newdata = bi_test, # testing data
                          type = "class")

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# testing dataset (test).
RF_ttest_conf <- confusionMatrix(data = tune.teRFpreds, # predictions
                                 reference = bi_test$sentiment_dependent, # actual
                                 positive = "positive",
                                 mode = "everything")
RF_ttest_conf 


## Goodness of Fit

# To assess if the model is balanced,
# underfitting or overfitting, we compare
# the performance on the training and
# testing. We can use the cbind() function
# to compare side-by-side.

# Overall
cbind(Training = RF_ttrain_conf$overall,
      Testing = RF_ttest_conf$overall)

# Class-Level
cbind(Training = RF_ttrain_conf$byClass,
      Testing = RF_ttest_conf$byClass)


# D.1.3: Feature Selection

# Feature Selection
#Let's try feature selection to reduce our model training time, and its complexity for better
#interpretation and possible improved accuracy.

#Let's first subset the most important variables.
FS_Vars <-importance(x  = rf_mod)
FS_Vars

# Subsetting top 20 best predictors base on Gini Index
FS_20 <- rownames(as.data.frame(FS_Vars) %>% arrange(-MeanDecreaseGini))[1:20]

# Analysis
set.seed(2)
rf_FS <- randomForest(formula = sentiment_dependent~ . ,
                      data = bi_train%>% select(sentiment_dependent,all_of(FS_20)),
                      importance = TRUE,
                      ntree = 500)
# View the basic output

rf_FS

# Training Performance
FS.RFpreds <- predict(object = rf_FS,
                      type = "class")

RF_FS_conf <- confusionMatrix(data = FS.RFpreds,
                              reference = bi_train$sentiment_dependent,
                              positive = "positive",
                              mode = "everything")

RF_FS_conf


#Testing Performance

FS.teRFpreds <- predict(object = rf_FS,
                        newdata = bi_test,
                        type = "class")

RF_btest_FS_conf <- confusionMatrix(data = FS.teRFpreds,
                                    reference = bi_test$sentiment_dependent,
                                    positive = "positive",
                                    mode = "everything")

RF_btest_FS_conf


# Performance is roughly the same as the base model

# Goodness of fit

options(scipen = 30)
# Overall
cbind(Training = RF_FS_conf$overall,
      Testing = RF_btest_FS_conf$overall)
options(scipen = 9)

# Class-Level
cbind(Training = RF_FS_conf$byClass,
      Testing = RF_btest_FS_conf$byClass)

#D.1.4: Utilizing Random Forest Probability Calculation To Keep A Bigger Sample Size.

# In this section, we will try to keep all the 3 classes in the data for testing since removing all neutral
# rows may make our sample size too small (876 obs. compared to 1300 obs.)

# We still want to keep 3 classes for the testing set.
data_3classes <- bi_stockDf[complete.cases(bi_stockDf), ]

# Let's predict on the testing data with 3 classes and use probability mode.

base.teRf3class <- predict(object = rf_mod, # RF Base model
                           newdata = data_3classes, # testing data
                           type = "prob" )#probability mode instead of class mode

View(base.teRf3class)


#Let's set a threshold for our predictions. After trying various combinations, we believe the threshold
# should be: If the Positive % Column has the following percentage of being accurately predicted:
      # 0-25%: Negative Class
      # 25%-85%: Neutral Class
      # 85% - 100%: Positive Class

base.teRf3class <- as.data.frame(base.teRf3class) %>%
                    mutate(prediction=ifelse(positive > 0.85, "positive", 
                                             ifelse(positive < 0.25, "negative","neutral")))

View(base.teRf3class)


#The Confusion Matrix Performance
base.teRf3class$prediction <- as.factor(base.teRf3class$prediction)
data_3classes$sentiment_dependent <- as.factor(data_3classes$sentiment_dependent)
RF_CFM_3class <- confusionMatrix(data = base.teRf3class$prediction, # predictions
                                 reference = stock_data_frame$sentiment_dependent, # actual original
                                 positive = "positive",
                                 mode = "everything")

RF_CFM_3class$overall
RF_CFM_3class$byClass


# As anticipated, due to the 3 classes in the target variable, the performance can't be as good as the base 
#model.



# D.2: Lexicons


### Apply the pre-processing function to my_data and save the output to my_data_clean
pre_processing_data_frame_fct <- function(text_column){
  text_column <- tolower(text_column) # bring all the words to lower case
  text_column <- gsub('[[:digit:]]','',text_column) # remove numbers
  text_column <- gsub(paste(stopwords('en'), collapse = '\\b|\\b'),'',text_column) # remove stopwords
  text_column <- gsub('[[:punct:]]','',text_column) # remove punctuation
  text_column <- gsub('\\s+',' ',text_column) # remove white space
  text_column <- lemmatize_strings(text_column) # lemmatize text
  corp <- Corpus(VectorSource(text_column)) # transform to corpus
  return(corp)
}


my_data_clean <- pre_processing_data_frame_fct(tweets_labelled$text)
my_data_clean

# transform the clean data into a term document matrix
my_tdm <- TermDocumentMatrix(my_data_clean)



#  this gives a row for each term-document combination and the number of times each term appears in each document
tidy_frame <- tidy(my_tdm)
head(tidy_frame)
str(tidy_frame) #


# Categorical sentiment assignment:
# bing
sentiment_bing <- get_sentiments("bing")
sentiment_bing

# loughran
sentiment_loughran <- get_sentiments("loughran") 
sentiment_loughran

# Numerical value sentiment assignment:
# afinn
sentiment_afinn <- get_sentiments("afinn") 
sentiment_afinn

### change the format of the data frames that store the values of the sentiments to later merge them
# bing
sentiment_bing <- sentiment_bing %>% rename(score_bing = sentiment) %>% 
  mutate(score_bing = ifelse(score_bing == "negative", -1, 1))
head(sentiment_bing)
tail(sentiment_bing)
#loughran
unique(sentiment_loughran$sentiment)
sentiment_loughran <- sentiment_loughran %>% rename(score_loughran = sentiment) %>% 
  filter(score_loughran %in% c("negative", "positive")) %>% 
  mutate(score_loughran = ifelse(score_loughran == "negative", -1, 1))
head(sentiment_loughran)
tail(sentiment_loughran)
# afinn
sentiment_afinn <- sentiment_afinn %>% rename(score_afinn = value)
head(sentiment_afinn)
tail(sentiment_afinn)

### put the scores from all 3 dictionaries in 1 data frame
sentiments <- full_join(sentiment_bing, sentiment_loughran, by = c("word" = "word"))
sentiments <- full_join(sentiments, sentiment_afinn, by = c("word" = "word"))
head(sentiments)
tail(sentiments)

# you can use the tidy frame to bring the sentiment to it
# the tidy_frame and sentiment_bing data sets have the term and word columns in common
head(tidy_frame)
head(sentiments)

#merge the tidy_frame and sentiment_bing data sets; use by.x = "term" and by.y = "word to indicate the column 
# that is the same in both data sets. Keep all the rows from tidy_frame and bring the matching results from 
# the sentiment_bing data set
my_sentiments <- left_join(tidy_frame, sentiments, by = c("term" = "word"))
my_sentiments <- my_sentiments %>% arrange(document, term) # sort the data by the document and term columns
head(my_sentiments)
str(my_sentiments)

# bring the sentiment column to my_sentiments
# in the initial dataset, tweets_labelled, create a column that represents the row number; 
# this equals the number of the documents and we will use this column to be able to bring the sentiment
# column to the data frame that  has the sentiment scores
tweets_labelled <- tweets_labelled %>% mutate(document = row_number()) 
my_sentiments <- my_sentiments %>% mutate(document = as.integer(document))
my_sentiments <- full_join(my_sentiments, tweets_labelled %>% select(document, sentiment),
                           by = c("document" = "document"))

### Replace all NAs in the dictionary columns with 0
my_sentiments <- my_sentiments %>% mutate_at(vars(score_bing, score_loughran, score_afinn),  ~ if_else(is.na(.), 0, .))
head(my_sentiments)


### Create a wordcloud
# to create a wordcloud, we need to know the list of words and how many times each word shows up
# to do so, we can use the term and count columns from my_sentiments data 
# currently, my_sentiments data, shows the counts of words for each document
# we nee the total counts of each word, so let's sum count for each term
cloud_data <- my_sentiments %>% group_by(term) %>% summarise(counts = sum(count))
cloud_data <- cloud_data %>% filter(!is.na(term))  # there is an NA in the term column; let's remove this row, otherwise you will have issues plotting the wordcloud
head(cloud_data %>% arrange(-counts))  # these are the most common words
wordcloud(words=cloud_data$term, freq=cloud_data$counts, random.order=FALSE, colors=brewer.pal(7, "Greens"), max.words = 70, min.freq = 20)

### Calculate the lexicon scores per document per word; 
# multiply the number of times each word shows up in a document by the score of each dictionary
my_sentiments <- my_sentiments %>% mutate(score_bing = count * score_bing,
                                          score_loughran = count * score_loughran,
                                          score_afinn = count * score_afinn)
head(my_sentiments)

# as of now, we have a score for each of the 3 dictionaries per word-document combination
# what we need is a score for each dictionary for  each document
# to find this out, we need to sum up the scores per document
my_sentiments <- my_sentiments %>% group_by(document, sentiment) %>% 
  summarise(sum_score_bing = sum(score_bing),
            sum_score_loughran = sum(score_loughran),
            sum_score_afinn = sum(score_afinn))
head(my_sentiments)



head(my_sentiments)
tail(my_sentiments)

## As we did with ML Classification Models, We can also remove neutral class for better performance.


my_sentiments_2 <- my_sentiments

sentiment_lex <- my_sentiments_2[(my_sentiments_2$sentiment == "positive" | my_sentiments_2$sentiment == "negative"),] 
sentiment_lex_full<- sentiment_lex[complete.cases(sentiment_lex), ]

# Since sentiments still have 3 classes, we should reconvert it to factor so that R knows
# we only need 2 classes
sentiment_lex_full$sentiment <- as.character(sentiment_lex_full$sentiment)
sentiment_lex_full$sentiment <- as.factor(sentiment_lex_full$sentiment)

View(sentiment_lex_full)

#We would want to subset the my_sentiments dataframe to extract the 3 columns with predicted sentiments.
sents_sub_2 <- sentiment_lex_full[ ,(ncol(sentiment_lex_full)-2):ncol(sentiment_lex_full)]
sents_sub_2 <- sents_sub_2 %>% mutate(sum_score_bing = ifelse(sum_score_bing >= 0, "positive", "negative")) %>%
  mutate(sum_score_loughran = ifelse(sum_score_loughran >= 0, "positive", "negative")) %>%
  mutate(sum_score_afinn = ifelse(sum_score_afinn >= 0, "positive", "negative"))
sents_sub_2 <- data.frame(lapply(X = sents_sub_2, 
                                 FUN = as.factor))
#Loughran's performance
loughran_cm_2 <- confusionMatrix(data = sents_sub_2$sum_score_loughran, reference = sentiment_lex_full$sentiment , positive = "positive", mode = "everything")
loughran_cm_2

# Bing's performance
bing_cm_2 <- confusionMatrix(data = sents_sub_2$sum_score_bing, reference = sentiment_lex_full$sentiment , positive = "positive", mode = "everything")
bing_cm_2

#Afinn performance
afinn_cm_2 <- confusionMatrix(data = sents_sub_2$sum_score_afinn, reference = sentiment_lex_full$sentiment , positive = "positive", mode = "everything")
afinn_cm_2

# do the same for bing and afinn. Need to convert tweets_labelled' neutral sentiments to positive too.


#Comparing 3 lexicons
cbind(loughran = loughran_cm_2$overall,
      bing = bing_cm_2$overall,
      afinn = afinn_cm_2$overall)

cbind(loughran = loughran_cm_2$byClass,
      bing = bing_cm_2$byClass,
      afinn = afinn_cm_2$byClass)

