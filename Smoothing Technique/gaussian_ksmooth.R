# 1-d Gaussian kernel smoothing

# h is kernel radius/half bandwidth
gs_kernel_smoother <- function(x_vector, y_vector, h){
  
  kernel_weighted_avg <- c()
    
  for (i in 1:length(x_vector)) {
    x_star <- x_vector[i]
    neighbor_index <- which(x_vector >= x_star - h & x_vector <= x_star + h )
    y_nb <- y_vector[neighbor_index]
    x_nb <- x_vector[neighbor_index]
    
    kernel_value <- exp(-1/2 * (abs(x_nb - x_star)/h)^2)
    kernel_weight <- kernel_value/sum(kernel_value)
    kernel_weighted_avg[i] <- sum(y_nb * kernel_weight) 
  }
  
  return(kernel_weighted_avg)
}


