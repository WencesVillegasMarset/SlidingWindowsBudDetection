#rm(list=ls())
library(e1071)

patchClassifier <- function(descriptor) {
    descriptor<-data.frame(t(descriptor))
    colnames(descriptor) <- NULL
    load("/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/s25_r1_it1_csvm_bestmodel.RData")
    pred <- predict(c_svm_bestmodel, descriptor, probability=TRUE)
    prob <- attr(pred, "probabilities")
    print(prob[1])
    
    ## TODO: DEBERIA SER UN SOLO VALOR!
}