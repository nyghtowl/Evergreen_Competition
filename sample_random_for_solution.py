# - Dated  : 19-August-2013 Karachi, Pakistan. 
# - Kaggle : StumbleUpon Evergreen Classification Challenge
# - Benchmark starter code by Afroz Hussain (aliafroz@hotmail.com)
# - Training method : randomForest in R by using CARET Package , Edited in Notepad++

rm(list=ls(all=TRUE))

NUMBER_OF_TREES <- 500
NUMBER_OF_VARIABLES_TO_TRY <- 8
NUMBER_OF_CV <- 10
REPEATES <- 3

require(caret)
require(pROC)

cat ("loading training data...\n")
train <- read.table("train.tsv", header = T, sep = "\t")

cat ("pre-processing training data ...\n")
# - Pre Process Training dataset
    
    # -- Remove columns 
    train <- subset(train, select = -c(url, urlid, is_news, framebased, boilerplate ))
    
    # -- Set label as factor to specity classification problem
    train$label <- as.factor(train$label)
    levels(train$label) <- list(class_1 = "1", class_0 = "0")
    
    # -- Encode Categorical Feature
    train$alchemy_category = as.character(train$alchemy_category)
    
    #train$alchemy_category[ train$alchemy_category == "?" ] <- "0"
    train$alchemy_category [ train$alchemy_category  == "unknown" ] <- "0" 
    train$alchemy_category [ train$alchemy_category  == "arts_entertainment" ] <- "1"
    train$alchemy_category [ train$alchemy_category  == "business" ] <- "2"
    train$alchemy_category [ train$alchemy_category  == "computer_internet" ] <- "3"
    train$alchemy_category [ train$alchemy_category  == "culture_politics" ] <- "4"
    train$alchemy_category [ train$alchemy_category  == "gaming" ] <- "5"
    train$alchemy_category [ train$alchemy_category  == "health" ] <- "6"
    train$alchemy_category [ train$alchemy_category  == "law_crime" ] <- "7"
    train$alchemy_category [ train$alchemy_category  == "recreation" ] <- "8"
    train$alchemy_category [ train$alchemy_category  == "religion" ] <- "9"
    train$alchemy_category [ train$alchemy_category  == "science_technology" ] <- "10"
    train$alchemy_category [ train$alchemy_category  == "sports" ] <- "11"
    train$alchemy_category [ train$alchemy_category  == "weather" ] <- "12"

    train$alchemy_category <- as.numeric(train$alchemy_category)
    
    train$alchemy_category_score = as.character(train$alchemy_category_score)   
    train$alchemy_category_score [ train$alchemy_category_score == "?" ] <- "0.400001"  
    train$alchemy_category_score <- as.numeric(train$alchemy_category_score)    
    
    train$news_front_page  <- as.character(train$news_front_page )
    train$news_front_page [ train$news_front_page == "?" ] = "0.5"
    train$news_front_page  <- as.numeric(train$news_front_page )
    
        

set.seed(78692110)  
    
cat ("training...\n")

rfGrid <- expand.grid(.mtry = NUMBER_OF_VARIABLES_TO_TRY )

fitControl <- trainControl(
    method = "repeatedcv",
    number = NUMBER_OF_CV, 
    repeats = REPEATES,
    classProb = TRUE,
    summaryFunction = twoClassSummary)
    
    
model <- train( 
    label~.,
    data = train,
    method ="rf",
    trControl = fitControl,
    ntree = NUMBER_OF_TREES,
    importance = TRUE,  
    tuneGrid = rfGrid,
    metric = "ROC") 
        

# - Predict on provided test data set

cat("loading test data...")
test <- read.table("test.tsv", header = T, sep = "\t")
    # -- Pre Process test data
    urlid <- test$urlid
    test <- subset(test, select = -c(url, urlid, is_news, framebased, boilerplate ))
    
    test$alchemy_category = as.character(test$alchemy_category)

    test$alchemy_category[ test$alchemy_category == "?" ] <- "0"
    test$alchemy_category[test$alchemy_category  == "unknown"] <- "0" 
    test$alchemy_category[test$alchemy_category  == "arts_entertainment"] <- "1"
    test$alchemy_category[test$alchemy_category  == "business"] <- "2"
    test$alchemy_category[test$alchemy_category  == "computer_internet"] <- "3"
    test$alchemy_category[test$alchemy_category  == "culture_politics"] <- "4"
    test$alchemy_category[test$alchemy_category  == "gaming"] <- "5"
    test$alchemy_category[test$alchemy_category  == "health"] <- "6"
    test$alchemy_category[test$alchemy_category  == "law_crime"] <- "7"
    test$alchemy_category[test$alchemy_category  == "recreation"] <- "8"
    test$alchemy_category[test$alchemy_category  == "religion"] <- "9"
    test$alchemy_category[test$alchemy_category  == "science_technology"] <- "10"
    test$alchemy_category[test$alchemy_category  == "sports"] <- "11"
    test$alchemy_category[test$alchemy_category  == "weather"] <- "12"

    test$alchemy_category = as.numeric(test$alchemy_category)   
    test$alchemy_category_score = as.character(test$alchemy_category_score) 
    test$alchemy_category_score [ test$alchemy_category_score == "?" ] <- "0.400001"    
    test$alchemy_category_score = as.numeric(test$alchemy_category_score)   
    
    test$news_front_page  = as.character(test$news_front_page )
    test$news_front_page [ test$news_front_page == "?" ] = "0.5"
    test$news_front_page  = as.numeric(test$news_front_page )

    cat ("predicting ...\n")
    
pred <- predict(model, newdata = test, type = "prob")   
submit <- data.frame(urlid, pred$class_1)
names(submit)[2] <- "label"

write.csv(submit, "submit.csv", row.names = F)
cat("model result:\n")
print (model)   