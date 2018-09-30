################
##
## INSTRUCTIONS:
## The following file performs refitted lasso regression with k-fold cross-validation
## 
################


lassoISTA <- function(X, Y, lambda){
  
  THRESH = 1e-22
  p = ncol(X)
  n = nrow(X)
  L = eigen(t(X) %*% X)$values[1]
  
  XtX = t(X) %*% X
  XtY = t(X) %*% Y
  
  beta = rep(0, p)
  while (TRUE){
    beta_new = softThresh( beta - (1/L)*(2/n)*(XtX %*% beta - XtY), lambda/L)
    if (sum((beta_new - beta)^2 < THRESH))
      break
    else
      beta = beta_new
  }
  
  return(beta_new)
}



softThresh <- function(u, lambda){
  u[abs(u) <= lambda] = 0
  u[u > lambda] = u[u > lambda] - lambda
  u[u < -lambda] = u[u < -lambda] + lambda
  return(u)
}


cars = read.csv("cars.csv", as.is=TRUE)
set.seed(1)

Y = as.vector(cars[, "mpg"])
X = as.matrix(cars[, !(names(cars) %in% c("mpg", "name"))])
oldX = scale(X)
old_p = ncol(oldX)
## We create interaction features of the form
## column j * column j' for all (j, j')
## column j * (column j')^2 for all (j, j')
## (column j)^2 * (column j')^2 for all (j, j')
for (j in 1:old_p){
  X = cbind(X, oldX*oldX[, j], oldX*oldX[, j]^2, oldX^2*oldX[, j], oldX^2 * oldX[,j]^2)
}

Y = scale(Y)
X = scale(X)
X = cbind(X, rep(1, nrow(X)))

n = 200
test_ix = sample(nrow(X), nrow(X) - n)
X1 = X[-test_ix, ]
X2 = X[test_ix, ]
Y1 = Y[-test_ix]
Y2 = Y[test_ix]

p = ncol(X1)
n = nrow(X1)

K = 10

## randomly permute the rows of X1
set.seed(1)
new_order <- sample(nrow(X1), nrow(X1))
X1 <- X1[new_order, ]
Y1 <- Y1[new_order]

lambda_ls = 10^(seq(-2, 1, 0.05))

errs = rep(0, length(lambda_ls))

for (k in 1:K){
  valid_ix = ((k-1)*(n/K) + 1):(k*(n/K))
  
  Xtrain <- X1[-valid_ix, ]
  Ytrain <- Y1[-valid_ix]
  Xvalid <- X1[valid_ix, ]
  Yvalid <- Y1[valid_ix]
  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]
    ## compute lasso estimate with ISTA
    beta_lasso =  lassoISTA(Xtrain, Ytrain, lambda)  
    
    S = which(abs(beta_lasso) > 1e-10)
    
    if (length(S) == 0)
      errs[il] = Inf
    else {
      XS = Xtrain[, S]
      ## For refitting, we use ridge regression with a small penalty instead of 
      ## OLS in the event that the columns of X are not linearly independent
      beta_refit = solve(t(XS) %*% XS + 1e-10 * diag(length(S)), t(XS) %*% Ytrain)
      errs[il] =  errs[il] + sum((Yvalid - Xvalid[, S] %*% beta_refit)^2) 
    }
  }
}

lambda_star = lambda_ls[which.min(errs)]

beta_lasso = lassoISTA(X1, Y1, lambda_star)

S = which(abs(beta_lasso) > 1e-10)

## compute the refitting on X1, Y1
beta_refit =  solve( t(X1[, S]) %*% X1[, S] +  1e-10 * diag(length(S)), t(X1[, S]) %*% Y1 )    
test_error =  mean( (Y2 - X2[, S] %*% beta_refit)^2)  
  
## For comparison, we also compute the ridge 
beta_ols = solve(t(X1) %*% X1 + 1e-10 * diag(ncol(X1)), t(X1) %*% Y1)
ols_error = mean((X2 %*% beta_ols - Y2)^2)

## compute ridge where we only use the first 7 variables and 
## the all 1 constant feature.
S = c(1:7, ncol(X1))
beta_ols2 = solve(t(X1[, S]) %*% X1[, S] + 1e-10 * diag(length(S)), t(X1[, S]) %*% Y1)
ols2_error = mean((X2[, S] %*% beta_ols2 - Y2)^2)

baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   OLS: %.3f   OLS (with first 7 vars): %.3f", 
              test_error, baseline, ols_error, ols2_error))
