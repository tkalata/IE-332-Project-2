library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

install_tensorflow(extra_packages="pillow")
install_keras()

install.packages('imager')
library(imager)


#angelica test
#rotation install
#install.packages("imager")
#rotation lib
#library(imager)

#rotation image
#img <- load.image("Dandelion+Pictures9-1654859321.jpg")
#img_rotated <- rotate(img, angle = 45) # rotate by 45 degrees
#save.image(img_rotated, "Dandelion+Pictures9-1654859321_rotated.jpg")
#readJPEG


setwd("C:/Users/15135/Downloads/IE332 Project 2")
model<-load_model_tf("./dandelion_model")

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


install.packages('imager')
library(imager)

# Load images
test_image <- image_load(paste("./dandelions/",1,sep=""),
                         target_size = c(224,224))
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255


matrix <- c(0.7, 0.3, 0,
            -0.3, 0.7, 0,
            0, 0, 1)

# Apply affine transformation to each image
deformed_images <- lapply(f, function(x) {
  affine(x, matrix, output.dim = dim(x))
})

# Save deformed images
save.image.list(deformed_images, "path/to/deformed_images/")
