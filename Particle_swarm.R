library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

#DEFINE MODEL
model<-load_model_tf("./dandelion_model")


#Algorithm

#Particle Swarm algorithm
library(psoptim)

#attrition is the vector that says yes or no about dandelion
#uses random forest algorithm to train model on subset of data
fit_func <- function(x, attrition){
  a <- round(x[1])
  b <- round(x[2])
  
  model_spec <- rand_forest(mode="classification", mtry=a, min_n=b, trees=500)
  
  model_spec <- set_engine(model_spec, engine="ranger", num.threads = parallel::detectCores(), importance="impurity")
  
  #model fitting
  train_image <- image_load(paste("./grass/",i,sep=""), target_size = c(224,224)) 
                 + image_load(paste("./dandelions/",i,sep=""), target_size = c(224,224))
  model <- fit_xy(object=model_spec, x=select(data_train, -attrition), y=select(data_train, attrition))
  
  #use subsection of data to test model to return accuracy
  pred_test <- predict(model, new_data= data_test %>% select(-attrition))
  acc <- accuracy_vec(truth = data_test$attrition, estimate=pred_test$.pred_class)
  return(acc)
}

psoptim(par=rep(NA,2), fn= fit_func, lower=c(1,1), upper=c(17,128),
        control=list(maxit=1000, maxit.stagnate=10, vectorize=T, s=100))


#basic iterative method
library(torch)
library(torchvision)

attack_BIM <- function(mean, std, model, image, class_index, epsilon, alpha, iterations){
  
  #convert label to torch tensor of shape
  class_index <- torch.tensor(class_index)
  
  #check input image and label shape/size
  assert(image.shape == torch.Size(c(1,2,224,224)))
  assert(class_index.shape == torch.Size(c(1)))
  
  #initialize adversarial image
  image_adver <- image.clone()
  
  #calculate normalized range [0,1] and convert to tensors
  zero_normed <- c((1-m)/s for m,s in zip(mean,std))
  zero_normed <- torch.tensor(zero_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
  
  max_normed <- c((1-m)/s for m, s in zip(mean,std))
  max_normed <- torch.tensor(max_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
  
  #calculate normalized alpha
  alpha_normed <- alpha/s for s in std
  alpha_normed <- torch.tensor(alpha_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
  
  #calculated normalized epsilon and convert to tensor
  eps_normed <- epsilon/s for s in std
  eps_normed <- torch.tensor(eps_normed, dtype=torch.float)
  
  #calculate maximum change in pixel 
  image_plus <- image + eps_normed
  image_minus <- image - eps_normed
  
  i <- 1
  for(i <= iterations){
    image_adver <- image_adver.clone().detach()
    image_adver.requires_grad <- TRUE
    
    pred <- model(image_adver)
    loss <- F.nll_loss(pred, class_index)
    model.zero_gard()
    loss.backward()
    grad_x <- image_adver.grad.data
    
    #check if gradient exists
    assert(image_adver.grad != NA)
    
    #calculate x_prime
    image_prime <- image_adver + alpha_normed * grad_x.detach().sign()
    assert(torch.equal(image_prime, image_adver) == FALSE)
    
    part1 <- torch.max(image_minus, image_prime)
    part2 <- torch.max(zero_normed, part1)
    
    image_adver <- torch.min(image_plus, part2)
    image_adver <- torch.min(max_normed, image_adver)
  }
  return(image_adver)
}


#TESTING
res=c("","")
f=list.files("./grass")
for (i in f){
  test_image <- image_load(paste("./grass/",i,sep=""),
                           target_size = c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  #x <- color(x)
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    print(i)
  }
  print(pred[1,2])
}


res=c("","")
f=list.files("./dandelions")
for (i in f){
  test_image <- image_load(paste("./dandelions/",i,sep=""),
                           target_size = c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  #x <- color(x)
  pred <- model %>% predict(x)
  if(pred[1,1]<0.50){
    print(i)
  }
  print(pred[1,1])
}
print(res)

