library(glmnet)

lasso_loop <- function(fold_index, model_index) {
  ##----------------------------------------------------------------------------
  ## Configuration
  ## num_iterations: Number of iterations. Appearance would be either equal to
  ##                 or less than the number of iterations.
  ## input_path    : location of an input file
  ## save_path     : location for files to be saved
  ## train_csv_name: Filename of demographic information containing (K - 1)
  ##                 training fold (where K = 10)
  ##----------------------------------------------------------------------------
  num_iterations = 100
  input_path = 'C:/Users/jmkwon/headache'
  save_path = 'C:/Users/jmkwon/headache'
  train_csv_name = sprintf("train_%d.csv", fold_index)
  ##----------------------------------------------------------------------------
  if (fold_index < 1 || fold_index > 10) {
    stop("invalid fold index")
  }
  if (model_index < 1 || model_index > 3) {
    stop("invalid model case")
  }
  cont_features <- c('age', 'severity')
  output_name <- 'Y'

  csv_path <- paste(input_path, csv_name, sep='/')
  df <- read.csv(csv_path, header=TRUE)

  ## set ID as rownames
  rownames(df) <- df[, 1]
  df <- df[, -1]

  ## preprocessing
  df[, 'age'] <- 0.02 * df[, 'age'] - 1
  df[, 'severity'] <- 0.2 * df[, 'severity'] - 1

  y_true <- subset(df, select=c(output_name))
  rownames(y_true) <- rownames(df)
  x_raw <- df[, !(colnames(df) %in% c(output_name))]
  x_cont <- x_raw[, cont_features]#subset(x_raw, select=cont_features)
  x_cat <- x_raw[, !(colnames(x_raw) %in% cont_features)]

  if (model_index == 1) {
    y_test <- (y_true == 1)
    xfactors <- model.matrix(y_test~., data=x_cat)
    x <- as.matrix(data.frame(x_cont, xfactors))
  } else if (model_index == 2) {
    mask = !(y_true %in% c(1))
    y_test <- y_true[mask] == 2
    x_second_cont <- x_cont[mask,]
    x_second_cat <- x_cat[mask,]
    xfactors <- model.matrix(y_test~., data=x_second_cat)
    x <- as.matrix(data.frame(x_second_cont, xfactors))
  } else if (model_index == 3) {
    mask = !(y_true %in% c(1, 2))
    y_test <- y_true[mask] == 4
    x_second_cont <- x_cont[mask,]
    x_second_cat <- x_cat[mask,]
    xfactors <- model.matrix(y_test~., data=x_second_cat)
    x <- as.matrix(data.frame(x_second_cont, xfactors))
  }
  total.coef <- read.csv(text='occurrence')
  for(i in 1:num_iterations) {
    set.seed(i * 500)
    cv.lasso <- cv.glmnet(x, y_test, family='binomial', nfolds=10, alpha=1, standardize=FALSE)
    lasso.coef = predict(cv.lasso, type='coefficients', s=cv.lasso$lambda.1se)
    print(sprintf('[LASSO] iterating %d over %d... [%.2f%s]', i, num_iterations,
                  (i / num_iterations) * 100, '%'))
    for(features in lasso.coef@Dimnames[[1]][lasso.coef@i + 1]) {
      if (length(which(rownames(total.coef) == features)) == 0) {
        total.coef[features,] <- 1
      } else {
        total.coef[features,] <- total.coef[features,] + 1
      }
    }
  }
  print(total.coef)
  write.csv(total.coef, paste(save_path, sprintf('model_%d_fold_%d.csv', model_index, fold_index), sep='/'))
}

for(model_index in 1:3) {
  for(fold_index in 1:10) {
    lasso_loop(fold_index, model_index)
  }
}
