install_tensorflow(extra_packages="pillow")
install_keras()
install_tensorflow()


#import libraries
library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
library(jpeg)

#set up model
setwd("C:/Users/grant/Downloads/model/dandelion_model")
model<-load_model_tf("C:/Users/grant/Downloads/model/dandelion_model")

#insert algorithm here

## occlusion- add random pixels to file. Call this function in the test image?

occlude <-function(imgA){
  #imgA is name of image, collapse that with the filepath?
  path <- "C:/Users/grant/Downloads/model/dandelion_model"
  newimg<- paste(path, imgA, sep="" , collapse = "" )
  newimg<- str_replace_all(string=newimg, pattern=" ", repl="")
  img <- readJPEG(newimg) 
  dims <- dim(img)
  numPix <- dims[1] * dims[2]
  budget <- floor(numPix/100)
  #make code that adds random pixels spaced out at pixel budget
  
}



#### TEST

res=c(224,224)
f=list.files("./grass")
for (i in f){
  imgA <- paste("/grass/", i,sep="", collapse ="")
  test_image <- image_load(paste("./grass/",i,sep=""),
                           target_size =c(224,224))
  
  occlude(imgA)
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    print(i)
  }
}

res=c(224,224)
f=list.files("./dandelions")
for (i in f){
  test_image <- image_load(paste("./dandelions/",i,sep=""),
                           target_size =c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,1]<0.50){
    print(i)
  }
}
print(res)
