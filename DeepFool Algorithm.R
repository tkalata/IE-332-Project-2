#DeepFool by:
#LTS4. (2017, August 24). DeepFool/deepfool.py at master · LTS4/deepfool. GitHub. Retrieved April 27, 2023, from https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py 

#allowing python
install.packages("keras")
library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
# R's torch
library(torch)

install_tensorflow(extra_packages="pillow")
install_keras()

#beginning of algorithm
deepfool <- function(image, net, num_classes=10, overshoot=0.02, max_iter=50) {
  
  # Check if GPU is available
  is_cuda <- torch_is_available() && torch_device() == "cuda"
  
  if (is_cuda) {
    cat("Using GPU\n")
    image <- image$cuda()
    net <- net$cuda()
  } else {
    cat("Using CPU\n")
  }
  
  f_image <- net$forward(Variable(image[None, , , ], requires_grad = TRUE))$data$cpu()$numpy()$flatten()
  I <- (np_array(f_image))$flatten()$argsort(descending = TRUE)
  
  I <- I[1:num_classes]
  label <- I[1]
  
  input_shape <- dim(image)
  pert_image <- image$clone()
  w <- np_zeros(input_shape)
  r_tot <- np_zeros(input_shape)
  
  loop_i <- 0
  
  x <- Variable(pert_image[None, , , ], requires_grad = TRUE)
  fs <- net$forward(x)
  fs_list <- list()
  for (k in 1:num_classes) {
    fs_list[[k]] <- fs[1, I[k]]$item()
  }
  k_i <- label
  
  while (k_i == label && loop_i < max_iter) {
    
    pert <- Inf
    fs[1, I[1]]$backward(retain_graph = TRUE)
    grad_orig <- x$grad$data$cpu()$numpy()$copy()
    
    for (k in 2:num_classes) {
      zero_gradients(x)
      
      fs[1, I[k]]$backward(retain_graph = TRUE)
      cur_grad <- x$grad$data$cpu()$numpy()$copy()
      
      # set new w_k and new f_k
      w_k <- cur_grad - grad_orig
      f_k <- (fs[1, I[k]] - fs[1, I[1]])$data$cpu()$numpy()
      
      pert_k <- abs(f_k) / norm(w_k, "F")
      
      # determine which w_k to use
      if (pert_k < pert) {
        pert <- pert_k
        w <- w_k
      }
    }
    
    # compute r_i and r_tot
    # Added 1e-4 for numerical stability
    r_i <- (pert + 1e-4) * w / norm(w, "F")
    r_tot <- r_tot + r_i
    
    if (is_cuda) {
      pert_image <- image + (1 + overshoot) * torch_from_numpy(r_tot)$cuda()
    } else {
      pert_image <- image + (1 + overshoot) * torch_from_numpy(r_tot)
    }
    
    x <- Variable(pert_image, requires_grad = TRUE)
    fs <- net$forward(x)
    k_i <- fs$data$cpu()$numpy()$flatten()$which.max()
    
    loop_i <- loop_i + 1
  }
  
  r_tot <- (1 + overshoot) * r_tot
  
  return(list(r_tot, loop_i, label, k_i, pert_image))
}
