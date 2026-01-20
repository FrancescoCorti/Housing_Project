library(plm)
library(lmtest)
library(dplyr)
library(ivreg)
library(modelsummary)
library(tibble)
library(car)

############################################

# LOAD THE DATA

# PROVINCE
df <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model.csv')

pdata <- pdata.frame(df, index=c("prov","year"))


############################################


# CONTROL TESTS


# UNIT-ROOT TESTS - BUY MAX
# Levin-Lin-Chu -  
# The test procedures are designed to evaluate the null hypothesis that each individual in the panel 
# has integrated time series versus the alternative hypothesis that all individuals time series are stationary.

purtest(
  pdata$log_prov_buy_max_wgt,
  test = "levinlin",
  exo = "intercept",
  lags = 1
)

# Im-Pesaran-Shin
# It also allows for some (but not all) of the individual series to have unit roots under the alternative hypothesis.
# H1: a satisfactory number (stationary/unit root = non zero) of individual processes is stationary 
purtest(
  pdata$log_prov_buy_max_wgt,
  test = "ips",
  exo = "intercept",
  lags = 1
)

# Maddala-Wu
# The idea behind this test is to break up the hypothesis H0 : ri à 0 for alli, i à 1, 2, . . . , N into a set of 
# sub-hypotheses H0i: ri à 0 and noting thatH0 is wrong if and only if any of its components H0i is wrong.
purtest(
  pdata$log_prov_buy_max_wgt,
  test = "madwu",
  exo = "intercept",
  lags = 1
)


# UNIT-ROOT TESTS - BUY MIN
# Levin-Lin-Chu -  
# The test procedures are designed to evaluate the null hypothesis that each individual in the panel 
# has integrated time series versus the alternative hypothesis that all individuals time series are stationary.

purtest(
  pdata$log_prov_buy_min_wgt,
  test = "levinlin",
  exo = "intercept",
  lags = 1
)

# Im-Pesaran-Shin
# It also allows for some (but not all) of the individual series to have unit roots under the alternative hypothesis.
# H1: a satisfactory number (stationary/unit root = non zero) of individual processes is stationary 
purtest(
  pdata$log_prov_buy_min_wgt,
  test = "ips",
  exo = "intercept",
  lags = 1
)

# Maddala-Wu
# The idea behind this test is to break up the hypothesis H0 : ri à 0 for alli, i à 1, 2, . . . , N into a set of 
# sub-hypotheses H0i: ri à 0 and noting thatH0 is wrong if and only if any of its components H0i is wrong.
purtest(
  pdata$log_prov_buy_min_wgt,
  test = "madwu",
  exo = "intercept",
  lags = 1
)

# VIF - MAIN MODEL
vif <- plm(
  log_prov_buy_max_wgt ~
    log_tourism_index_rnk +
    log_prov_median_income_wgt +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(vif)


# VIF - NO INDEX
vif <- plm(
  log_prov_buy_max_wgt ~
    log_prov_ratio_tot_beds +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
    data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(vif)


##########################################################

# OLS FIXED-EFFECT

# PROVINCE FE (rnk) - min
fe_model_rnk_min <- plm(
  log_prov_buy_min_wgt ~
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_rnk_min)

# PROVINCE FE (rnk) - max 
fe_model_rnk_max <- plm(
  log_prov_buy_max_wgt ~
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_rnk_max)

###########################################################################

# PROVINCE FE (z) - min
fe_model_z_min <- plm(
  log_prov_buy_min_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_z_min)

# PROVINCE FE (z) - max 
fe_model_z_max <- plm(
  log_prov_buy_max_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_z_max)

#######################################################################


# PROVINCE FE (no index) - min
fe_model_no_min <- plm(
  log_prov_buy_min_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_no_min)

# PROVINCE FE (no index) - max 
fe_model_no_max <- plm(
  log_prov_buy_max_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_no_max)


##############################################################################

# measure persistency

pool_ar1 <- plm(log_prov_buy_max_wgt ~ lag(log_prov_buy_max_wgt, 1),
                data = pdata,
                model = "pooling")

summary(pool_ar1)

fe_ar1 <- plm(log_prov_buy_max_wgt ~ lag(log_prov_buy_max_wgt, 1),
                data = pdata,
                model = "within")

summary(fe_ar1)

##########################################################################################################

# SYSTEM-GMM - rnk

# PROVINCE - system-GMM - rnk - min
sys_model_rnk_min <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_min, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - rnk - max
sys_model_rnk_max <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_max, robust = TRUE, time.dummies = FALSE)


#######################################################################

# SYSTEM-GMM - z

# PROVINCE - system-GMM - z - min
sys_model_z_min <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_min, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - z - max
sys_model_z_max <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_max, robust = TRUE, time.dummies = FALSE)


####################################################################à

# PROVINCE - system-GMM (LEVEL) - NO INDEX - min
sys_model_no_min <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_min, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM (LEVEL) - NO INDEX - max
sys_model_no_max <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_max, robust = TRUE, time.dummies = FALSE)

##################################################################################################

# SYSTEM-GMM COVID

# PROVINCE - system-GMM - rnk - max - covid
sys_model_rnk_max_covid <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid:tourism_index_rnk |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(covid:tourism_index_rnk, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_max_covid, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - z - max - covid
sys_model_z_max_covid <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid:tourism_index_z |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(covid:tourism_index_z, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_max_covid, robust = TRUE, time.dummies = FALSE)



# PROVINCE - system-GMM (LEVEL) - NO INDEX - max - covid
sys_model_no_max_covid <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid:log_prov_ratio_tot_nights +
    covid:log_prov_ratio_str_houses |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(covid:log_prov_ratio_tot_nights, 2:10) +
    lag(covid:log_prov_ratio_str_houses, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_max_covid, robust = TRUE, time.dummies = FALSE)





# SYSTEM-GMM COVID - MIN

# PROVINCE - system-GMM - rnk - min - covid
sys_model_rnk_min_covid <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid:tourism_index_rnk |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(covid:tourism_index_rnk, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_min_covid, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - z - min - covid
sys_model_z_min_covid <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid:tourism_index_z |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(covid:tourism_index_z, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_min_covid, robust = TRUE, time.dummies = FALSE)



# PROVINCE - system-GMM (LEVEL) - NO INDEX - min - covid
sys_model_no_min_covid <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid:log_prov_ratio_tot_nights +
    covid:log_prov_ratio_str_houses |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(covid:log_prov_ratio_tot_nights, 2:10) +
    lag(covid:log_prov_ratio_str_houses, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_min_covid, robust = TRUE, time.dummies = FALSE)


