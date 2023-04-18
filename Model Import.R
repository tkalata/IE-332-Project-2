#PACKAGE INSTALLATION

#install.packages("keras")
#install.packages()

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
#install_tensorflow(extra_packages="pillow")
#install_keras()

#DEFINE MODEL
model<-load_model_tf("./dandelion_model")


#ALGORITHMS

#noise install
install.packages("jpeg")


#noise lib
library(jpeg)


#reduce res

#img <- readJPEG("636597665741397587-dandelion-1097518082.jpg")
#img_noise <- img + rnorm(length(img), sd=0.8) # add random noise
#writeJPEG(img_noise, "636597665741397587-dandelion-1097518082_noisy.jpg")

f=list.files("./grass")
f1 <- paste("./grass/",f,sep="")
reduce_res <- function(f){
  #test_image <- image_load("./grass/")
  for (i in f){
    img <- readJPEG(f[i])
    img_noise <- img + rnorm(length(img), sd=0.8) # add random noise
    noise_vec[i] <- writeJPEG(img_noise, target=raw())
  }
}


#rescale

resizePixels <- function(image, w=150,h=150){
  pixels <- as.vector(image)
  #initial width/height
  w1 = nrow(image)
  h1 = ncol(image)
  #create empty vector
  temp <- vector('numeric', w*h)
  #compute ratios
  x_ratio <- w1/w
  y_ratio <- h1/h
  #resizing
  for (i in 0: (h-1)) {
    for (j in 0: (w-1)) {
      px <- floor(j*x_ratio)
      py <- floor(i*y_ratio)
      temp[(i*w)+j] <- pixels[(py*w1)+px]
    }
  }
  m <- matrix(temp,h,w)
  return(m)
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
  pred <- model %>% predict(x)
  if(pred[1,1]<0.50){
    print(i)
  }
}
print(res)


