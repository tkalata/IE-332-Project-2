install.packages("keras")
library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

install_tensorflow(extra_packages="pillow")
install_keras()


#noise install
install.packages("jpeg")


#noise lib
library(jpeg)

#setwd("C:/Users/angel/OneDrive/Documents/IE 332 Project 2")
model<-load_model_tf("./dandelion_model")



#reduce resolution
#image: Dandelion+Pictures9-1654859321
img <- readJPEG("Dandelion+Pictures9-1654859321.jpg")
img_noise <- img + rnorm(length(img), sd=0.1) # add random noise
writeJPEG(img_noise, "Dandelion+Pictures9-1654859321_noisy.jpg")



# model test

res=c("","")
f=list.files("./grass")
for (i in f){
  test_image <- image_load(paste("./grass/",i,sep=""),
                           target_size =  c(224,224))
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









