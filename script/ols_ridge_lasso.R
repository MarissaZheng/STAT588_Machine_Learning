cars = read.csv("cars.csv", as.is=TRUE)

Y = as.vector(cars[, "mpg"])
X = as.matrix(cars[, !(names(cars) %in% c("mpg", "name"))])

# can optionally standardize X and Y for OLS and ridge, but must for LASSO so coeeficients are regualized on the same scale
Y = scale(Y)
X = scale(X)
num_trash_vars = 100  ## try changing this

X = cbind(X, matrix(rnorm(num_trash_vars * nrow(X)), nrow(X), num_trash_vars))
X = cbind(X, rep(1, nrow(X)))

# divide into training and testing data set
set.seed(1)
train_ix = sample(nrow(X), floor(nrow(X)/2))
test_ix = (1:nrow(X))[-train_ix]

Xtrain = X[train_ix, ]
Xtest = X[test_ix, ]

Ytrain = Y[train_ix]
Ytest = Y[test_ix]

# OLS
beta_ols = solve(t(Xtrain) %*% Xtrain, t(Xtrain) %*% Ytrain)
ols_in_samp_err = mean((Ytrain - Xtrain %*% beta_ols)^2)
ols_out_samp_err = mean((Ytest - Xtest %*% beta_ols)^2)

# ridge regualarization
p = ncol(Xtrain)
lambda = 100   ## Try changing this
beta_ridge = solve(t(Xtrain) %*% Xtrain + lambda * diag(p), t(Xtrain) %*% Ytrain)
ridge_in_samp_err = mean((Ytrain - Xtrain %*% beta_ridge)^2)
ridge_out_samp_err = mean((Ytest - Xtest %*% beta_ridge)^2)

print(sprintf("OLS:  in-sample error is %.2f,  out-of-sample error is %.2f", ols_in_samp_err, ols_out_samp_err))
print(sprintf("Ridge:  in-sample error is %.2f,  out-of-sample error is %.2f", ridge_in_samp_err, ridge_out_samp_err))


#lasso
source("ISTA.R")

lambda_ls = 0.000037  ## Try changing this, more sensitive to grid than lambda_ridge 

beta_lasso = lassoISTA(Xtrain, Ytrain, lambda_ls)


# compare plot
dt <- data.frame(feature = 1:ncol(Xtrain), beta_ols = beta_ols, beta_ridge = beta_ridge, beta_lasso = beta_lasso)
dt_long <- melt.data.frame(dt, id.vars = "feature", measure.vars= c(2:4), variable.name = "method", value.name = "beta")
colnames(dt_long) <- c("feature", "method", "beta")
p1 <- ggplot(data = dt_long) + 
  geom_point(aes(y = beta, x=feature, colour = method)) + 
  scale_color_manual(values = c("beta_ols" = "#ff00ff","beta_ridge" = "#3399ff", "beta_lasso" = "#f7290f" ))
  
  
p1
