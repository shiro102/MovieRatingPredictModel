library(rvest)
library(stringr)
library(tidyverse)
library(dslabs)
library(dplyr)
library(caret)
library(data.table)
library(gridExtra)

## Split edx into training and test data using same strategy as course split
# Create train set and test set
set.seed(1)
tempIndex <- createDataPartition(y = edx$rating, p=0.2, list = FALSE)
trainSet <- edx[-tempIndex,]
temp <- edx[tempIndex,]
# Make sure userId and movieId in validation set are also in edx set
testSet <- temp %>% 
  semi_join(trainSet, by = "movieId") %>%
  semi_join(trainSet, by = "userId")
# Add rows removed from validation set back into edx set
removed <- anti_join(temp, testSet)
trainSet <- rbind(trainSet, removed)
rm(tempIndex, temp, removed)

### Data Processing

#First we try to analyze some basic characteristics of the data set
trainSet %>% group_by(rating) %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black") + ggtitle("Rating Hist")
#As we can see, the rating distribution follows normal distribution with rating 3 and 4 have the highest use. In addition, we see that the 
#number of user who put ratings as integer is siginificantly higher than the ones who put ratings as decimal. For training purpose, we will
#only use the data that has rating is an integer in both trainSet and testSet
newTestSet <- testSet %>% filter(rating %% 1 == 0)
newTrainSet <- trainSet %>% filter(rating %% 1 == 0)
#We want to make a new graph for the new set
newTrainSet %>% group_by(rating) %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 1, color = "black") + ggtitle("New Rating Hist")


### Model Development

#Calculate mean of the new train set
mu <- mean(newTrainSet$rating)
mean_RMSE <- RMSE(testSet$rating, mu)
mean_accuracy <-  sum(newTestSet$rating == round(mu)) / length(newTestSet$rating)
#Calculate median of the new train set
med <- median(newTrainSet$rating)
median_RMSE <- RMSE(testSet$rating, med)
median_accuracy <- sum(newTestSet$rating == med) / length(newTestSet$rating)
#Create reporting data frame
reportedTable <- data.frame(Type = c("Mean", "Median"), Accuracy = c(mean_accuracy,median_accuracy), RMSE = c(mean_RMSE,median_RMSE))
reportedTable
#From previous data, we notice that if we only use mean as a tool for prediction, the accuracy is around 36%. The same goes for median

## Next, we try to apply more bias terms to improve our prediction, which is b_m (movie bias), b_u (user bias) and b_g (genre bias)
#Calculate movie bias b_m
b_mData <- newTrainSet %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu))
#Calculate user bias b_u
b_uData <- newTrainSet %>%
  left_join(b_mData, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))
#Calculate genre bias b_g
b_gData <- newTrainSet %>%
  left_join(b_mData, by='movieId') %>%
  left_join(b_uData, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_m - b_u))
#Combine all bias into new the prediction
predictRating <- newTestSet %>%
  left_join(b_mData, by='movieId') %>%
  left_join(b_uData, by='userId') %>%
  left_join(b_gData, by='genres') %>%
  mutate(prediction = mu + b_m + b_u + b_g) %>%
  mutate(roundedPrediction = round(prediction))
#Remove columns that have NA data due to loss of some certain movie when do data processing
naList <-which(is.na(newTestSet$rating == predictRating$roundedPrediction))
predictRating <- predictRating[-naList,]
newTestSet <- newTestSet[-naList,]
# Calculate new accuracy and RMSE
predict_mean_accuracy <- sum( predictRating$roundedPrediction == newTestSet$rating) / length(newTestSet$rating)
predict_RMSE <- RMSE(newTestSet$rating, predictRating$prediction)

###Regularization
#We try to remove datas that only have few ratings and have relatively high difference compare to the mean (Third-rated movie with only few ratings)
lambdas <- seq(1, 7, 0.1)
RMSEs_lambda <- sapply(lambdas, function(l){
  b_m <- newTrainSet %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  b_u <- newTrainSet %>%
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  
  b_g <- newTrainSet %>%
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_m - b_u - mu)/(n()+l))
  predicted_ratings <- newTestSet %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by="genres") %>%
    mutate(prediction = mu + b_m + b_u + b_g) %>%
    mutate(roundedPrediction = round(prediction))
  return(RMSE(newTestSet$rating, predicted_ratings$prediction))
})

RMSEfigure <- ggplot() + aes(lambdas, RMSEs_lambda) + geom_point() + xlab('Lambda') + ylab("RMSE") + ggtitle("Lambda Tuning")
RMSEfigure
max_lambda <- lambdas[which.min(RMSEs_lambda)]

#Recalculate using max_lambda
b_m <- newTrainSet %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+max_lambda))
b_u <- newTrainSet %>%
  left_join(b_m, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - mu)/(n()+max_lambda))
b_g <- newTrainSet %>%
  left_join(b_m, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_m - b_u - mu)/(n()+max_lambda))
predictRating <- newTestSet %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by="genres") %>%
  mutate(prediction = mu + b_m + b_u + b_g) %>%
  mutate(roundedPrediction = round(prediction))
# Calculate new accuracy and RMSE
new_predict_mean_accuracy <- sum( predictRating$roundedPrediction == newTestSet$rating) / length(newTestSet$rating)
new_predict_RMSE <- RMSE(newTestSet$rating, predictRating$prediction)

##Some more info graphs
#Errors graph
errorData <- data.frame(Error = c(predictRating$roundedPrediction - newTestSet$rating))
errorData
Errorfig <- errorData %>% ggplot(aes(Error))  + geom_histogram(binwidth = 1, color = "black") +
  ggtitle("Error Range") + xlab("Error") + ylab("Count")
Errorfig

#Figure compare between predicted and true rating
temp <- predictRating %>% group_by(rating) %>%
  ggplot() +
  geom_histogram(aes(roundedPrediction), binwidth = 0.5, fill="blue", alpha=0.5) + 
  ggtitle("Predict Rating") + xlab("Rating") + ylab("Count")
temp1 <- predictRating %>% group_by(rating) %>%
  ggplot() +
  geom_histogram(aes(rating), binwidth = 0.5, fill="red", alpha=0.5) + 
  ggtitle("True Rating") + xlab("Rating") + ylab("Count") +
  scale_y_continuous(limits=c(0, 700000))
grid.arrange(temp,temp1,nrow=1)

#Apply to validation set
predictRating <- validation %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by="genres") %>%
  mutate(prediction = mu + b_m + b_u + b_g) %>%
  mutate(roundedPrediction = round(prediction))
naList <-which(is.na(validation$rating == predictRating$roundedPrediction))
predictRating <- predictRating[-naList,]
validation <- validation[-naList,]
validation <- validation %>% filter(rating %% 1 == 0)
predictRating <- predictRating %>% filter(rating %% 1 == 0)

# Calculate validation accuracy and RMSE
valid_predict_mean_accuracy <- sum( predictRating$roundedPrediction == predictRating$rating) / length(predictRating$rating)
valid_predict_RMSE <- RMSE(validation$rating, predictRating$prediction)
