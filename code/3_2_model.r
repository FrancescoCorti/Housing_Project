library(plm)
library(lmtest)
library(dplyr)
library(ivreg)
library(modelsummary)
library(tibble)
library(car)



library(sf)
library(spdep)
library(spatialreg)

############################################

# LOAD THE DATA

# PROVINCE
df <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model.csv')

pdata <- pdata.frame(df, index=c("prov","year"))


############################################


# CONTROL TESTS


# UNIT-ROOT TESTS - BUY AVG
# Levin-Lin-Chu -  
# The test procedures are designed to evaluate the null hypothesis that each individual in the panel 
# has integrated time series versus the alternative hypothesis that all individuals time series are stationary.

purtest(
  pdata$log_prov_buy_avg_wgt,
  test = "levinlin",
  exo = "intercept",
  lags = 1
)

# Im-Pesaran-Shin
# It also allows for some (but not all) of the individual series to have unit roots under the alternative hypothesis.
# H1: a satisfactory number (stationary/unit root = non zero) of individual processes is stationary 
purtest(
  pdata$log_prov_buy_avg_wgt,
  test = "ips",
  exo = "intercept",
  lags = 1
)

# Maddala-Wu
# The idea behind this test is to break up the hypothesis H0 : ri à 0 for alli, i à 1, 2, . . . , N into a set of 
# sub-hypotheses H0i: ri à 0 and noting thatH0 is wrong if and only if any of its components H0i is wrong.
purtest(
  pdata$log_prov_buy_avg_wgt,
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

# VIF - MAIN MODEL
vif <- plm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_rnk +
    log_prov_median_income_wgt +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index,
  data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(vif)


# VIF - NO INDEX
vif <- plm(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_beds +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index,
    data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(vif)


##########################################################

# OLS FIXED-EFFECT


# PROVINCE FE (rnk) - avg
fe_model_rnk_avg <- plm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_rnk_avg)


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
    log_interest_rate +
    log_cpi_index,
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
    log_interest_rate +
    log_cpi_index,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_rnk_max)

###########################################################################

# PROVINCE FE (z) - avg
fe_model_z_avg <- plm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_z_avg)


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
    log_interest_rate +
    log_cpi_index,
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
    log_interest_rate +
    log_cpi_index,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_z_max)

#######################################################################


# PROVINCE FE (no index) - avg
fe_model_no_avg <- plm(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_no_avg)


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
    log_interest_rate +
    log_cpi_index,
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
    log_interest_rate +
    log_cpi_index,
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

# PROVINCE - system-GMM - rnk - avg
sys_model_rnk_avg <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index|
    lag(log_prov_buy_avg_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_avg, robust = TRUE, time.dummies = FALSE)


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
    log_interest_rate +
    log_cpi_index|
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
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
    log_interest_rate +
    log_cpi_index|
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
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

# PROVINCE - system-GMM - z - avg
sys_model_z_avg <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_avg, robust = TRUE, time.dummies = FALSE)


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
    log_interest_rate +
    log_cpi_index |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
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
    log_interest_rate +
    log_cpi_index |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_max, robust = TRUE, time.dummies = FALSE)
####################################################################à

# PROVINCE - system-GMM (LEVEL) - NO INDEX - avg
sys_model_no_avg <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    #log_prov_ratio_tot_beds +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:10) +
    #lag(log_prov_ratio_tot_beds, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_avg, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM (LEVEL) - NO INDEX - min
sys_model_no_min <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    #log_prov_ratio_tot_beds +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index |
    lag(log_prov_buy_min_wgt, 2:10) +
    #lag(log_prov_ratio_tot_beds, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
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
    #log_prov_ratio_tot_beds +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    log_cpi_index |
    lag(log_prov_buy_max_wgt, 2:10) +
    #lag(log_prov_ratio_tot_beds, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 2:10) +
    lag(log_interest_rate, 2:10) +
    lag(log_cpi_index, 2:10),
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


# PROVINCE - system-GMM - rnk - avg - covid
sys_model_rnk_avg_covid <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid_all +
    covid_all:tourism_index_rnk +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_tourism_index_rnk, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 2:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(log_real_gdp, 2:7) +
    lag(log_interest_rate, 2:7) +
    lag(covid_all:tourism_index_rnk, 2:7) +
    lag(log_cpi_index, 2:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_avg_covid, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - z - avg - covid
sys_model_z_avg_covid <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid_all +
    covid_all:tourism_index_z +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_tourism_index_z, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 2:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(log_real_gdp, 2:7) +
    lag(log_interest_rate, 2:7) +
    lag(covid_all:tourism_index_z, 2:7) +
    lag(log_cpi_index, 2:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_avg_covid, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM (LEVEL) - NO INDEX - avg - covid
sys_model_no_avg_covid <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid_all +
    covid_all:log_prov_ratio_tot_nights +
    covid_all:log_prov_ratio_str_houses +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_prov_ratio_tot_nights, 2:7) +
    lag(log_prov_ratio_str_houses, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 2:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(log_real_gdp, 2:7) +
    lag(log_interest_rate, 2:7) +
    lag(covid_all:log_prov_ratio_tot_nights, 2:7) +
    lag(covid_all:log_prov_ratio_str_houses, 2:7) +
    lag(log_cpi_index, 2:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_avg_covid, robust = TRUE, time.dummies = FALSE)

#########################################################

# COVID - 2020-2021 AND 2022

# PROVINCE - system-GMM - rnk - avg - covid
sys_model_rnk_avg_covid <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid_2022 +
    covid:tourism_index_rnk +
    covid_2022:tourism_index_rnk +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_tourism_index_rnk, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 2:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(log_real_gdp, 2:7) +
    lag(log_interest_rate, 2:7) +
    lag(covid:tourism_index_rnk, 2:7) +
    lag(covid_2022:tourism_index_rnk, 2:7) +
    lag(log_cpi_index, 2:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_avg_covid, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - z - avg - covid
sys_model_z_avg_covid <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid +
    covid_2022 +
    covid:tourism_index_z +
    covid_2022:tourism_index_z +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_tourism_index_z, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 2:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(log_real_gdp, 2:7) +
    lag(log_interest_rate, 2:7) +
    lag(covid:tourism_index_z, 2:7) +
    lag(covid_2022:tourism_index_z, 2:7) +
    lag(log_cpi_index, 2:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_avg_covid, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM (LEVEL) - NO INDEX - avg - covid
sys_model_no_avg_covid <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
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
    covid_2022 +
    covid:log_prov_ratio_tot_nights +
    covid:log_prov_ratio_str_houses +
    covid_2022:log_prov_ratio_tot_nights +
    covid_2022:log_prov_ratio_str_houses +
    log_cpi_index |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_prov_ratio_tot_nights, 2:7) +
    lag(log_prov_ratio_str_houses, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 2:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(log_real_gdp, 2:7) +
    lag(log_interest_rate, 2:7) +
    lag(covid:log_prov_ratio_tot_nights, 2:7) +
    lag(covid:log_prov_ratio_str_houses, 2:7) +
    lag(covid_2022:log_prov_ratio_tot_nights, 2:7) +
    lag(covid_2022:log_prov_ratio_str_houses, 2:7) +
    lag(log_cpi_index, 2:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_avg_covid, robust = TRUE, time.dummies = FALSE)

########################################################à

# DEFLATED ROBUST CHECK

# PROVINCE - system-GMM - rnk - avg - de
sys_model_rnk_avg_de <- pgmm(
  log_prov_buy_avg_wgt_de ~ 
    lag(log_prov_buy_avg_wgt_de, 1) +
    log_tourism_index_rnk +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt_de, 2:10) +
    lag(log_tourism_index_rnk, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_rnk_avg_de, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - z - avg
sys_model_z_avg_de <- pgmm(
  log_prov_buy_avg_wgt_de ~ 
    lag(log_prov_buy_avg_wgt_de, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt_de, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_avg_de, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM - no - avg
sys_model_no_avg_de <- pgmm(
  log_prov_buy_avg_wgt_de ~ 
    lag(log_prov_buy_avg_wgt_de, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt_de, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_no_avg_de, robust = TRUE, time.dummies = FALSE)


#####################################################################################

# SPATIAL REGRESSION

prov_sf <- st_read(
  "C:/Users/HP/Desktop/Traineeship/Code/code/datasets/mun_gis_data/prov_map_updated.shp"
)

# compute queen contiguity weights
nb_q <- poly2nb(prov_sf, queen = TRUE)

W_q <- nb2listw(nb_q, style = "W", zero.policy = TRUE)

# representative year
df_2019 <- pdata %>% filter(year == 2019)

# Moran I test
moran.test(df_2019$log_prov_buy_avg_wgt_de,
           listw = W_q,
           zero.policy = TRUE)

sp_2019 <- pdata %>%
  filter(year == 2019) %>%
  arrange(prov)   # MUST match prov_sf order

nrow(sp_2019)
length(W_q$neighbours)


# spatial lag model - rnk
sar_model_rnk <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_rnk +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_model_rnk)

# spatial lag model - z
sar_model_z <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_z +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_model_z)


# spatial lag model - no
sar_model_no <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_model_no)

############################################################

# spatial spillover of tourism - rnk
sp_df_rnk <- pdata %>%
  select(prov, year,
         log_prov_buy_avg_wgt_de,
         log_tourism_index_rnk,
         log_prov_population,
         log_prov_median_income_wgt,
         log_prov_unemployment)


sp_df_rnk <- sp_df_rnk %>%
  group_by(year) %>%
  mutate(W_tourism = lag.listw(W_q, log_tourism_index_rnk,
                               zero.policy = TRUE)) %>%
  ungroup()


spill_model_rnk <- lm(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_rnk +
    W_tourism +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    factor(year),
  data = sp_df_rnk
)

summary(spill_model_rnk)



# spatial spillover of tourism - z
sp_df_z <- pdata %>%
  select(prov, year,
         log_prov_buy_avg_wgt_de,
         log_tourism_index_z,
         log_prov_population,
         log_prov_median_income_wgt,
         log_prov_unemployment)


sp_df_z <- sp_df_z %>%
  group_by(year) %>%
  mutate(W_tourism = lag.listw(W_q, log_tourism_index_z,
                               zero.policy = TRUE)) %>%
  ungroup()


spill_model_z <- lm(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_z +
    W_tourism +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    factor(year),
  data = sp_df_z
)

summary(spill_model_z)


# spatial spillover of tourism - nights

sp_df_nights <- pdata %>%
  select(prov, year,
         log_prov_buy_avg_wgt_de,
         log_prov_ratio_tot_nights,
         log_prov_population,
         log_prov_median_income_wgt,
         log_prov_unemployment)


sp_df_nights <- sp_df_nights %>%
  group_by(year) %>%
  mutate(W_tourism = lag.listw(W_q, log_prov_ratio_tot_nights,
                               zero.policy = TRUE)) %>%
  ungroup()


spill_model_nights <- lm(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_tot_nights +
    W_tourism +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    factor(year),
  data = sp_df_nights
)

summary(spill_model_nights)



# spatial spillover of tourism - str

sp_df_str <- pdata %>%
  select(prov, year,
         log_prov_buy_avg_wgt_de,
         log_prov_ratio_str_houses,
         log_prov_population,
         log_prov_median_income_wgt,
         log_prov_unemployment)


sp_df_str <- sp_df_str %>%
  group_by(year) %>%
  mutate(W_tourism = lag.listw(W_q, log_prov_ratio_str_houses,
                               zero.policy = TRUE)) %>%
  ungroup()


spill_model_str <- lm(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_str_houses +
    W_tourism +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    factor(year),
  data = sp_df_str
)

summary(spill_model_str)



