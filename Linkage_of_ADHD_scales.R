library(tidyverse)
library(caret)
library(splines)
library(randomForest)
library(equate)

### Create a mocking dataset with similar structure to the original one for testing purposes.
set.seed(123)

n <- 2000  # sample size

# Basic variables
age <- sample(6:18, n, replace = TRUE)
gender <- sample(c("Male", "Female"), n, replace = TRUE)
subtype <- sample( c("Combined", "Inattentive", "Hyperactive-Impulsive"), n, replace = TRUE)

# BMI at baseline and endpoint
bmi.bas <- runif(n, min = 12, max = 50)
bmi_change <- rnorm(n, mean = 0, sd = 3) # We expect not big change of BMI through a trial
bmi.endp <- bmi.bas + bmi_change

rm(bmi_change)




# Baseline CGI-S & ADHD-RS-IV

rho_bas <- 0.53  # Based on the correlation of the appendix
Sigma_bas <- matrix(c(1, rho_bas, rho_bas, 1), 2)
latent_bas <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma_bas)


# CGI-S baseline (3–7); Restrict values because of eligibility criteria in baseline
cgis.bas <- cut(
  latent_bas[,1], breaks = quantile(latent_bas[,1], probs = seq(0, 1, length.out = 6)),
  labels = 3:7, include.lowest = TRUE)
cgis.bas <- as.numeric(as.character(cgis.bas))


# ADHD-RS baseline (24–54); Restrict values because of eligibility criteria in baseline
adhd.rs.bas <- scales::rescale(latent_bas[,2], to = c(24, 54))
adhd.rs.bas <- round(adhd.rs.bas)


# Endpoint CGI-S & ADHD-RS-IV

rho_endp <- 0.84  # Based on the correlation of the appendix
Sigma_endp <- matrix(c(1, rho_endp, rho_endp, 1), 2)
latent_endp <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma_endp)


# CGI-S endpoint (1–7)
cgis.endp <- cut(
  latent_endp[,1], breaks = quantile(latent_endp[,1], probs = seq(0, 1, length.out = 8)),
  labels = 1:7, include.lowest = TRUE)
cgis.endp <- as.numeric(as.character(cgis.endp))


# ADHD-RS baseline (24–54); Restrict values because of eligibility criteria in baseline
adhd.rs.endp <- scales::rescale(latent_endp[,2], to = c(0, 54))
adhd.rs.endp <- round(adhd.rs.endp)


# Assume that there is a 20% drop out
dropout <- rbinom(n, size = 1, prob = 0.20)  # 1 = dropout

# Missingness is in the endpoint variables
bmi.endp[dropout == 1] <- NA
cgis.endp[dropout == 1] <- NA
adhd.rs.endp[dropout == 1] <- NA


# Create the Dataset for analysis

data <- data.frame(age, gender, subtype, bmi.bas, bmi.endp, cgis.bas, cgis.endp, adhd.rs.bas, adhd.rs.endp)

rm(list=setdiff(ls(), "data"))



################################################################################
################################################################################
######################## Absolute score ########################################
################################################################################
################################################################################



########################################################################### 
####################### Univariable models ################################
########################################################################### 


baseline.data <- data %>% select(adhdrs = adhd.rs.bas, cgis = cgis.bas) %>% na.omit()
endpoint.data <- data %>% select(adhdrs = adhd.rs.endp, cgis = cgis.endp) %>% na.omit()

abscor.uni <- rbind(baseline.data, endpoint.data)
rm(baseline.data, endpoint.data)



########### Compare the transformation algorithms with 10-fold CV ##############

set.seed(120294)
folds <- createMultiFolds(abscor.uni$adhdrs, k = 10, times = 1)


results_uni <- lapply(folds, function(test_idx){
  
  train <- abscor.uni[-test_idx, ]
  test <- abscor.uni[test_idx, ]
  
  # Equipercentile linking
  link <- equate(as.freqtab(table(train$cgis)), 
                 as.freqtab(table(train$adhdrs)),
                 type = "equipercentile", boot = T, reps= 10, smooth = "loglin")$concordance %>%
    select(scale, yx) %>% rename(cgis = scale)
  
  test_eq <- test %>% left_join(link, by = "cgis")

  obs <- test$adhdrs
  pred_eq <- test_eq$yx
    
  
  # Linear regression model
  model_lm <- lm(adhdrs ~ cgis, data = train)
  pred_lm <- predict(model_lm, newdata = test)

  # Splines model
  model_spl <- lm(adhdrs ~ ns(cgis, df = 4), data = train)
  pred_spl <- predict(model_spl, newdata = test)
  
  # Random forest
  model_rf <- randomForest(adhdrs ~ cgis, data = train)
  pred_rf <- predict(model_rf, newdata = test)
  
  # Compare them in the same dataset for fairness (during splitting in equipercentile some values might
  # not exist in the validation dataset and this will lead to NA predictions for equipercentile)
  idx <- !is.na(pred_eq)

  obs_cc <- obs[idx]
  pred_eq_cc <- pred_eq[idx]
  pred_lm_cc <- pred_lm[idx]
  pred_spl_cc <- pred_spl[idx]  
  pred_rf_cc <- pred_rf[idx]
    
  
  # Measurement for model comparison (Root mean squared error, R^2 and Median absolute error)  
  metrics <- function(predicted, observed){
    
    rmse <- sqrt(mean((predicted - observed)^2))
    mae <- median(abs(predicted - observed))
    r2 <- summary(lm(predicted ~ observed))[["r.squared"]]
    c(RMSE = rmse, MAE = mae, R2 = r2) 
  }
  
  m_eq <- metrics(pred_eq_cc, obs_cc)
  m_lm <- metrics(pred_lm_cc, obs_cc)
  m_spl <- metrics(pred_spl_cc, obs_cc)
  m_rf <- metrics(pred_rf_cc, obs_cc)
  

  c(EQ_RMSE = m_eq[["RMSE"]], EQ_R2 = m_eq[["R2"]], EQ_MAE = m_eq[["MAE"]],
    LR_RMSE = m_lm[["RMSE"]], LR_R2 = m_lm[["R2"]], LR_MAE = m_lm[["MAE"]],
    SL_RMSE = m_spl[["RMSE"]], SR_R2 = m_spl[["R2"]], SR_MAE = m_spl[["MAE"]],
    RF_RMSE = m_rf[["RMSE"]], RF_R2 = m_rf[["R2"]], RF_MAE = m_rf[["MAE"]])
  
})

results_mat <- do.call(rbind, results_uni)
final_results_uni <- round(colMeans(results_mat), 2)
final_results_uni




########################################################################### 
##################### Multivariable models ################################
########################################################################### 


pred.bas <- data %>% select(age, gender, subtype, bmi = bmi.bas, adhdrs = adhd.rs.bas, cgis = cgis.bas) %>% na.omit()
pred.fin <- data %>% select(age, gender, subtype, bmi = bmi.endp, adhdrs = adhd.rs.endp, cgis = cgis.endp)%>% na.omit()

pred <- rbind(pred.bas, pred.fin)
rm(pred.bas, pred.fin)


set.seed(120294)
folds <- createMultiFolds(pred$adhdrs, k = 10, times = 1)


########### Compare the transformation algorithms with 10-fold CV ##############
results_mult <- lapply(folds, function(test_idx){
  
  train <- pred[-test_idx, ]
  test <- pred[test_idx, ]
  
  obs <- test$adhdrs
  
  # Linear regression model
  model_lm <- lm(adhdrs ~ cgis + age + gender + subtype + bmi, data = train)
  pred_lm <- predict(model_lm, newdata = test)
  
  # Splines model
  model_spl <- lm(adhdrs ~ ns(cgis, df = 4) + ns(age, df = 4) + gender + subtype + bmi, data = train)
  pred_spl <- predict(model_spl, newdata = test)
  
  # Random forest
  model_rf <- randomForest(adhdrs ~ cgis + age + gender + subtype + bmi, data = train, ntree = 500, mtry = 2)
  pred_rf <- predict(model_rf, newdata = test)
  

  # Measurement for model comparison  
  metrics <- function(predicted, observed){
    
    rmse <- sqrt(mean((predicted - observed)^2))
    mae <- median(abs(predicted - observed))
    r2 <- summary(lm(predicted ~ observed))[["r.squared"]]
    c(RMSE = rmse, MAE = mae, R2 = r2) 
  }
  
  m_lm <- metrics(pred_lm, obs)
  m_spl <- metrics(pred_spl, obs)
  m_rf <- metrics(pred_rf, obs)
  

  
  c(LR_RMSE = m_lm[["RMSE"]], LR_R2 = m_lm[["R2"]], LR_MAE = m_lm[["MAE"]],
    SL_RMSE = m_spl[["RMSE"]], SR_R2 = m_spl[["R2"]], SR_MAE = m_spl[["MAE"]],
    RF_RMSE = m_rf[["RMSE"]], RF_R2 = m_rf[["R2"]], RF_MAE = m_rf[["MAE"]]
  )
  
})

results_mat <- do.call(rbind, results_mult)
final_results_mult <- round(colMeans(results_mat), 2)
final_results_mult



rm(list=setdiff(ls(), c("data", "final_results_uni", "final_results_mult")))







################################################################################
################################################################################
############################ Change from baseline ##############################
################################################################################
################################################################################


########################################################################### 
####################### Univariable models ################################
########################################################################### 

data <- data %>% mutate(adhdrs.chng = adhd.rs.endp - adhd.rs.bas,
                        cgis.chng = cgis.endp - cgis.bas,
                        bmi.chng = bmi.endp - bmi.bas)

change <- data %>% select(cgis.chng, adhdrs.chng) %>% na.omit()

 


########### Compare the transformation algorithms with 10-fold CV ##############
set.seed(120294)
folds <- createMultiFolds(change$adhdrs.chng, k = 10, times = 1)


results_uni <- lapply(folds, function(test_idx){
  
  train <- change[-test_idx, ]
  test <- change[test_idx, ]
  
  # Equipercentile linking
  link <- equate(as.freqtab(table(train$cgis.chng)), 
                 as.freqtab(table(train$adhdrs.chng)),
                 type = "equipercentile", boot = T, reps= 10, smooth = "loglin")$concordance %>%
    select(scale, yx) %>% rename(cgis.chng = scale)
  
  test_eq <- test %>% left_join(link, by = "cgis.chng")
  
  obs <- test$adhdrs.chng
  pred_eq <- test_eq$yx
  
  
  # Linear regression model
  model_lm <- lm(adhdrs.chng ~ cgis.chng, data = train)
  pred_lm <- predict(model_lm, newdata = test)
  
  # Splines model
  model_spl <- lm(adhdrs.chng ~ ns(cgis.chng, df = 4), data = train)
  pred_spl <- predict(model_spl, newdata = test)
  
  # Random forest
  model_rf <- randomForest(adhdrs.chng ~ cgis.chng, data = train, ntree = 500, mtry = 2)
  pred_rf <- predict(model_rf, newdata = test)
  
  # Compare them in the same dataset for fairness
  idx <- !is.na(pred_eq)
  
  obs_cc <- obs[idx]
  pred_eq_cc <- pred_eq[idx]
  pred_lm_cc <- pred_lm[idx]
  pred_spl_cc <- pred_spl[idx]  
  pred_rf_cc <- pred_rf[idx]
  
  
  # Measurement for model comparison  
  metrics <- function(predicted, observed){
    
    rmse <- sqrt(mean((predicted - observed)^2))
    mae <- median(abs(predicted - observed))
    r2 <- summary(lm(predicted ~ observed))[["r.squared"]]
    c(RMSE = rmse, MAE = mae, R2 = r2) 
  }
  
  m_eq <- metrics(pred_eq_cc, obs_cc)
  m_lm <- metrics(pred_lm_cc, obs_cc)
  m_spl <- metrics(pred_spl_cc, obs_cc)
  m_rf <- metrics(pred_rf_cc, obs_cc)
  

  c(EQ_RMSE = m_eq[["RMSE"]], EQ_R2 = m_eq[["R2"]], EQ_MAE = m_eq[["MAE"]],
    LR_RMSE = m_lm[["RMSE"]], LR_R2 = m_lm[["R2"]], LR_MAE = m_lm[["MAE"]],
    SL_RMSE = m_spl[["RMSE"]], SR_R2 = m_spl[["R2"]], SR_MAE = m_spl[["MAE"]],
    RF_RMSE = m_rf[["RMSE"]], RF_R2 = m_rf[["R2"]], RF_MAE = m_rf[["MAE"]])
  
})

results_mat <- do.call(rbind, results_uni)
final_results_uni_cfb <- round(colMeans(results_mat), 2)
final_results_uni_cfb

rm(folds, results_mat, results_uni, change)






################################################################################
######################## Multivariable prediction ##############################
################################################################################

pred.chng <- data %>% select(age, gender, bmi.chng, adhdrs.chng, subtype, cgis.chng) %>% na.omit()


########### Compare the transformation algorithms with 10-fold CV ##############
set.seed(120294)
folds <- createMultiFolds(pred.chng$adhdrs.chng, k = 10, times = 1)


results_mult <- lapply(folds, function(test_idx){
  
  train <- pred.chng[-test_idx, ]
  test <- pred.chng[test_idx, ]
  
  obs <- test$adhdrs.chng
  
  # Linear regression model
  model_lm <- lm(adhdrs.chng ~ cgis.chng + age + gender + subtype + bmi.chng, data = train)
  pred_lm <- predict(model_lm, newdata = test)
  
  # Splines model
  model_spl <- lm(adhdrs.chng ~ ns(cgis.chng, df = 4) + ns(age, df = 4) + gender + subtype + bmi.chng, data = train)
  pred_spl <- predict(model_spl, newdata = test)
  
  # Random forest
  model_rf <- randomForest(adhdrs.chng ~ cgis.chng + age + gender + subtype + bmi.chng, data = train, ntree = 500, mtry = 2)
  pred_rf <- predict(model_rf, newdata = test)
  
  
  # Measurement for model comparison  
  metrics <- function(predicted, observed){
    
    rmse <- sqrt(mean((predicted - observed)^2))
    mae <- median(abs(predicted - observed))
    r2 <- summary(lm(predicted ~ observed))[["r.squared"]]
    c(RMSE = rmse, MAE = mae, R2 = r2) 
  }
  
  m_lm <- metrics(pred_lm, obs)
  m_spl <- metrics(pred_spl, obs)
  m_rf <- metrics(pred_rf, obs)
  
  
  
  c(LR_RMSE = m_lm[["RMSE"]], LR_R2 = m_lm[["R2"]], LR_MAE = m_lm[["MAE"]],
    SL_RMSE = m_spl[["RMSE"]], SR_R2 = m_spl[["R2"]], SR_MAE = m_spl[["MAE"]],
    RF_RMSE = m_rf[["RMSE"]], RF_R2 = m_rf[["R2"]], RF_MAE = m_rf[["MAE"]]
  )
  
})

results_mat <- do.call(rbind, results_mult)
final_results_mult_cfb <- round(colMeans(results_mat), 2)
final_results_mult_cfb

rm(folds, results_mat, results_mult)


