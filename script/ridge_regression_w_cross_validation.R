

################
##
## INSTRUCTIONS:
## The following file performs ridge regression with k-fold cross-validation
## 
################


set.seed(1)
movies = read.csv("movies_hw1.csv")

# Feature construction
movies$log_budget_sq <- movies$log_budget^2
movies$log_revenue_sq <- movies$log_revenue^2
movies$log_vote_count <- log(movies$vote_count + 1)
movies$Action.Adven <- as.numeric(movies$Action & movies$Adventure)
movies$Rom.Com <- as.numeric(movies$Romance & movies$Comedy)
movies$vote_budget <- movies$log_vote_count * movies$log_budget
movies$long <- ifelse(movies$runtime > 120, 1, 0)

# #of training samples
n = 300  
test_ix = sample(nrow(movies), nrow(movies) - n)
test_ix
## Exclude title and vote_average
X = cbind(as.matrix(movies[, !(names(movies) %in% c("vote_average", "title"))]), rep(1, nrow(movies)))  
Y = movies[, "vote_average"]

# trainning dataset which is used for k-fold CV
X1 = X[-test_ix, ]   
Y1 = Y[-test_ix]

# test dataset
X2 = X[test_ix, ]   
Y2 = Y[test_ix]

p = ncol(X)
K = 10

# randomly permute the rows of X1
set.seed(1)
new_order <- sample(nrow(X1), nrow(X1))
new_order
X1 <- X1[new_order, ]
Y1 <- Y1[new_order]
lambda_ls = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3)
errs = rep(0, length(lambda_ls))

for (k in 1:K){
  valid_ix = ((k-1)*(n/K) + 1):(k*(n/K))
  Xtrain <- X1[-valid_ix, ]
  Ytrain <- Y1[-valid_ix]
  Xvalid <- X1[valid_ix, ]
  Yvalid <- Y1[valid_ix]
  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]
    beta_ridge = solve(t(Xtrain) %*% Xtrain + lambda*diag(p), t(Xtrain) %*% Ytrain)
    errs[il] = errs[il] + sum((Yvalid - Xvalid %*% beta_ridge)^2)
  }
}

lambda_star = lambda_ls[which.min(errs)]
beta_ridge_final = solve(t(X1) %*% X1 + lambda_star*diag(p), t(X1) %*% Y1)


test_error = mean((X2 %*% beta_ridge_final - Y2)^2)
baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   lambda_star: %f", test_error, baseline, lambda_star))
