install.packages("tidyverse")
install.packages("keras")
install.packages("tensorflow")
install.packages("reticulate")



library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)


install_tensor(extra_packages="pillow")
install_keras()

library(keras)

# Load the pre-trained model
model <- load_model_tf("./dandelion_model")


# generate a random decision stump
generate_stump <- function() {
  # Select a random feature and threshold
  feature <- sample(1:2, 1) # choose between height and width of the image
  threshold <- runif(1, min = 0, max = 1)
  # Define a function to classify an image based on the decision stump
  classify <- function(image) {
    if (image[[feature]] < threshold) {
      return("grass")
    } else {
      return("dandelion")
    }
  }
  # Return the decision stump as a list containing the classify function and its parameters
  return(list(classify = classify, feature = feature, threshold = threshold))
}

# Define a function to evaluate the accuracy of a decision stump on a set of test images
evaluate_stump <- function(stump, images, labels) {
  predictions <- sapply(images, function(image) stump$classify(image))
  accuracy <- sum(predictions == labels) / length(labels)
  return(accuracy)
}

# Set up initial parameters and data
test_dir <- "./test" # directory containing the test images
test_images <- flow_images_from_directory(test_dir, target_size = c(224, 224), batch_size = 1, shuffle = FALSE)$samples
test_labels <- flow_images_from_directory(test_dir, target_size = c(224, 224), batch_size = 1, shuffle = FALSE)$classes
T_max <- 1 # initial temperature
T_min <- 0.01 # minimum temperature
cooling_rate <- 0.95 # cooling rate
n_iterations <- 1000 # maximum number of iterations

# Initialize current solution as a random decision stump
current_stump <- generate_stump()
current_accuracy <- evaluate_stump(current_stump, test_images, test_labels)
best_stump <- current_stump
best_accuracy <- current_accuracy

# Main simulated annealing loop
for (i in 1:n_iterations) {
  # Generate a candidate solution by perturbing the current decision stump
  candidate_stump <- generate_stump()
  candidate_accuracy <- evaluate_stump(candidate_stump, test_images, test_labels)
  # Determine whether to accept the candidate solution based on its accuracy and the current temperature
  delta_accuracy <- candidate_accuracy - current_accuracy
  if (delta_accuracy > 0 || runif(1, min = 0, max = 1) < exp(delta_accuracy / T_max)) {
    current_stump <- candidate_stump
    current_accuracy <- candidate_accuracy
    if (current_accuracy > best_accuracy) {
      best_stump <- current_stump
      best_accuracy <- current_accuracy
    }
  }
  # Cool the system according to the cooling schedule
  T_max <- T_max * cooling_rate
  if (T_max < T_min) break # terminate if temperature falls below minimum
}

# Return the best decision stump found
return(best_stump)




model <- load_model_tf("./dandelion_model")
res=c("","")
f=list.files("./grass")
for (i in f){
  test_image <- image_load(paste("./grass/",i,sep=""),
                           target_size = c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  x <- best_stump
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    print(i)
  }
}

res=c("","")
f=list.files("./dandelions")
for (i in f){
  test_image <- image_load(paste("./dandelions/",i,sep=""),
                           target_size = c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  x <- best_stump
  pred <- model %>% predict(x)
  if(pred[1,1]<0.50){
    print(i)
  }
}
print(res)
