library(tidyverse)
library(caret)
library(xgboost)
library(pROC)

train <- read.csv('train.csv')
test <- read.csv('test.csv')

train <- na.omit(train)
test <- na.omit(test)

train$churn <- as.numeric(as.factor(train$churn)) - 1

# Feature Engineering
train$interactionRate <- train$clicks / train$visits
test$interactionRate <- test$clicks / test$visits

# Standardization
# Note: Avoid standardizing the response variable 'churn' and any categorical variables
numeric_features <- c("clicks", "visits", "interactionRate") # Update with your actual numeric feature names
train[numeric_features] <- scale(train[numeric_features])
test[numeric_features] <- scale(test[numeric_features])

set.seed(123)
index <- createDataPartition(y = train$churn, p = 0.75, list = FALSE)
trainingData <- train[index, ]
validationData <- train[-index, ]

# Ensure the validation set churn variable is correctly formatted (if necessary)
validationData$churn <- as.numeric(as.factor(validationData$churn)) - 1

# Preparing matrices for XGBoost
train_matrix <- model.matrix(~ . -1, data = trainingData[,-which(names(trainingData) == "churn")])
label <- trainingData$churn
dtrain <- xgb.DMatrix(data = train_matrix, label = label)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Training the XGBoost model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, nthread = 1)

# Validation
validation_matrix <- model.matrix(~ . -1, data = validationData[,-which(names(validationData) == "churn")])
dvalidation <- xgb.DMatrix(data = validation_matrix)
validation_labels <- validationData$churn

validation_preds <- predict(xgb_model, dvalidation)
roc_result <- roc(validation_labels, validation_preds)
print(paste("Validation AUC Score:", roc_result$auc))

# Preparing for submission
test_matrix <- model.matrix(~ . -1, data = test)
dtest <- xgb.DMatrix(data = test_matrix)
test_preds <- predict(xgb_model, dtest)

submission <- data.frame(id = test$id, churn = test_preds)
write.csv(submission, 'xgb_submission_2.csv', row.names = FALSE)
library(ggplot2)

# Extract feature importance from the XGBoost model
importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)

# Convert the importance data to a data frame for easier manipulation
importance_df <- as.data.frame(importance)

# Plot feature importance vs. gain
ggplot(importance_df, aes(x = Feature, y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Feature Importance vs. Gain",
       x = "Feature",
       y = "Gain") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Churn Distribution Barplot
churn_counts <- table(train$churn)
barplot(churn_counts, names.arg = c("Not Churned", "Churned"), main = "Churn Distribution")

# Correlation Matrix
# Calculate correlation matrix
correlation_matrix <- cor(trainingData)

# Plot correlation matrix
corrplot::corrplot(correlation_matrix, method = "circle", type = "full", tl.cex = 0.7)

# Extract feature importance from the XGBoost model
importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)

# Convert the importance data to a data frame for easier manipulation
importance_df <- as.data.frame(importance)

# Plot feature importance vs. gain
ggplot(importance_df, aes(x = Feature, y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Feature Importance vs. Gain",
       x = "Feature",
       y = "Gain") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ROC Curve (assuming you've kept validation_preds and roc_result)
roc_curve <- roc(validation_labels, validation_preds)
plot(roc_curve, col = "blue", main = "ROC Curve", print.auc = TRUE)


