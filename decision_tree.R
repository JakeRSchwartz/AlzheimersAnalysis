install.packages("tree")

library(tree) 

dataset <- read.csv("alzheimers_disease_data.csv")

# display the structure of the dataset
str(dataset)

# provide summary statistics of the dataset
summary(dataset)

# remove rows with any NA values
dataset <- na.omit(dataset)

# convert necessary columns to factors
dataset$Gender <- as.factor(dataset$Gender)
dataset$Ethnicity <- as.factor(dataset$Ethnicity)
dataset$EducationLevel <- as.factor(dataset$EducationLevel)
dataset$Smoking <- as.factor(dataset$Smoking)
dataset$FamilyHistoryAlzheimers <- as.factor(dataset$FamilyHistoryAlzheimers)
dataset$CardiovascularDisease <- as.factor(dataset$CardiovascularDisease)
dataset$Diabetes <- as.factor(dataset$Diabetes)
dataset$Depression <- as.factor(dataset$Depression)
dataset$HeadInjury <- as.factor(dataset$HeadInjury)
dataset$Hypertension <- as.factor(dataset$Hypertension)
dataset$MemoryComplaints <- as.factor(dataset$MemoryComplaints)
dataset$BehavioralProblems <- as.factor(dataset$BehavioralProblems)
dataset$Confusion <- as.factor(dataset$Confusion)
dataset$Disorientation <- as.factor(dataset$Disorientation)
dataset$PersonalityChanges <- as.factor(dataset$PersonalityChanges)
dataset$DifficultyCompletingTasks <- as.factor(dataset$DifficultyCompletingTasks)
dataset$Forgetfulness <- as.factor(dataset$Forgetfulness)
dataset$Diagnosis <- as.factor(dataset$Diagnosis)

set.seed(1)

# create random sample of row indices
sample_index <- sample(seq_len(nrow(dataset)), size = 0.8 * nrow(dataset))

# assign 80% of the data to the training set
train_data <- dataset[sample_index, ]

# assign remaining 20% of the data to the test set
test_data <- dataset[-sample_index, ]

# build decision tree using training data
decision_tree <- tree(Diagnosis ~ ., data = train_data)

# perform cross-validation
cv_tree <- cv.tree(decision_tree, FUN = prune.misclass)

# prune tree
pruned_tree <- prune.misclass(decision_tree, best = cv_tree$size[which.min(cv_tree$dev)])

# plot pruned decision tree
plot(pruned_tree)
text(pruned_tree, pretty = 0)

# predict diagnosis on the test data
predictions <- predict(pruned_tree, test_data, type = "class")

# create confusion matrix
confusion_matrix <- table(test_data$Diagnosis, predictions)

# print confusion matrix
print(confusion_matrix)

# calculate accuracy of the model
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# print accuracy of the model
print(paste("accuracy:", accuracy))

# plot tree size vs deviance
plot(cv_tree$size, cv_tree$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")
 
#summary of base tree
summary(decision_tree)

#summary of prunned tree
summary(pruned_tree)

print(cv_tree)

