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


# CAPITALS
df_cap <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model_capital.csv')

pdata_cap <- pdata.frame(df_cap, index=c("mun_istat","year"))


############################################


# CONTROL TESTS


# UNIT-ROOT TESTS
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
  log_prov_buy_max_wgt ~
    log_tourism_index_vdw +
    log_prov_median_income_wgt +
    log_prov_population +
    log_prov_immigration +
    log_nominal_gdp +
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
    log_prov_median_income_wgt +
    log_prov_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate,
    data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(vif)


##########################################################

# OLS FIXED-EFFECT


# PROVINCE FE 
fe_model <- plm(
  log_prov_buy_max_wgt ~
    #lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_vdw +
    log_prov_median_income_wgt +
    log_prov_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model)

# CAPITALS FE 
fe_model_cap <- plm(
  log_buy_max ~
    #lag(log_buy_max, 1) +
    log_tourism_index_vdw +
    log_median_income +
    log_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate,
  data = pdata_cap,
  index = c("mun_istat", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_cap)


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

# SYSTEM-GMM LEVEL

# PROVINCE - system-GMM (LEVEL)
sys_model <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_vdw +
    log_prov_median_income_wgt +
    log_prov_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_vdw, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_nominal_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM (LEVEL) - NO INDEX
sys_model <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_prov_ratio_tot_beds +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_median_income_wgt +
    log_prov_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_prov_ratio_tot_beds, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_nominal_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model, robust = TRUE, time.dummies = FALSE)


# CAPITALS - system-GMM (LEVEL)
sys_model_cap <- pgmm(
  log_buy_max ~ 
    lag(log_buy_max, 1) +
    log_tourism_index_vdw +
    log_median_income +
    log_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_buy_max, 2:10) +
    lag(log_tourism_index_vdw, 2:10) +
    lag(log_median_income, 2:10) +
    lag(log_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_nominal_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata_cap,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("mun_istat", "year")
)

summary(sys_model_cap, robust = TRUE, time.dummies = FALSE)


# CAPITALS - system-GMM (LEVEL) - NO INDEX
sys_model_cap <- pgmm(
  log_buy_max ~ 
    lag(log_buy_max, 1) +
    log_ratio_tot_beds +
    log_ratio_tot_nights +
    log_ratio_str_houses +
    log_median_income +
    log_population +
    log_prov_immigration +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_buy_max, 2:10) +
    lag(log_ratio_tot_beds, 2:10) +
    lag(log_ratio_tot_nights, 2:10) +
    lag(log_ratio_str_houses, 2:10) +
    lag(log_median_income, 2:10) +
    lag(log_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_nominal_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata_cap,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("mun_istat", "year")
)

summary(sys_model_cap, robust = TRUE, time.dummies = FALSE)


##################################################################################################

# SYSTEM-GMM GROWTH

# PROVINCE GROWTH RATES
growth_data <- na.omit(pdata)

pdata1 <- pdata.frame(growth_data, index=c("prov","year"))

# PROVINCE - system-GMM (GROWTH)
sys_model_growth <- pgmm(
  d_log_prov_buy_max_wgt ~ 
    lag(d_log_prov_buy_max_wgt, 1) +
    d_log_tourism_index_vdw +
    d_log_prov_median_income_wgt +
    d_log_prov_population +
    d_log_prov_immigration +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_prov_buy_max_wgt, 2:10) +
    lag(d_log_tourism_index_vdw, 2:10) +
    lag(d_log_prov_median_income_wgt, 2:10) +
    lag(d_log_prov_population, 2:10) +
    lag(d_log_prov_immigration, 2:10) +
    lag(d_log_nominal_gdp, 2:10) +
    lag(d_log_interest_rate, 2:10),
  data = pdata1,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_growth, robust = TRUE, time.dummies = FALSE)


# PROVINCE - system-GMM (GROWTH) - NO INDEX
sys_model_growth_no <- pgmm(
  d_log_prov_buy_max_wgt ~ 
    lag(d_log_prov_buy_max_wgt, 1) +
    d_log_prov_ratio_tot_beds +
    d_log_prov_ratio_tot_nights +
    d_log_prov_ratio_str_houses +
    d_log_prov_median_income_wgt +
    d_log_prov_population +
    d_log_prov_immigration +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_prov_buy_max_wgt, 2:10) +
    lag(d_log_prov_ratio_tot_beds, 2:10) +
    lag(d_log_prov_ratio_tot_nights, 2:10) +
    lag(d_log_prov_ratio_str_houses, 2:10) +
    lag(d_log_prov_median_income_wgt, 2:10) +
    lag(d_log_prov_population, 2:10) +
    lag(d_log_prov_immigration, 2:10) +
    lag(d_log_nominal_gdp, 2:10) +
    lag(d_log_interest_rate, 2:10),
  data = pdata1,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_growth_no, robust = TRUE, time.dummies = FALSE)


# CAPITALS GROWTH RATES
growth_data_cap <- na.omit(pdata_cap)

pdata1_cap <- pdata.frame(growth_data_cap, index=c("mun_istat","year"))

# CAPITALS - system-GMM (GROWTH)
sys_model_growth_cap <- pgmm(
  d_log_buy_max ~ 
    lag(d_log_buy_max, 1) +
    d_log_tourism_index_vdw +
    d_log_median_income +
    d_log_population +
    d_log_prov_immigration +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_buy_max, 2:10) +
    lag(d_log_tourism_index_vdw, 2:10) +
    lag(d_log_median_income, 2:10) +
    lag(d_log_population, 2:10) +
    lag(d_log_prov_immigration, 2:10) +
    lag(d_log_nominal_gdp, 2:10) +
    lag(d_log_interest_rate, 2:10),
  data = pdata1_cap,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("mun_istat", "year")
)

summary(sys_model_growth_cap, robust = TRUE, time.dummies = FALSE)


# CAPITALS - system-GMM (GROWTH) - NO INDEX
sys_model_growth_cap_no <- pgmm(
  d_log_buy_max ~ 
    lag(d_log_buy_max, 1) +
    d_log_ratio_tot_beds +
    d_log_ratio_tot_nights +
    d_log_ratio_str_houses +
    d_log_median_income +
    d_log_population +
    d_log_prov_immigration +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_buy_max, 2:10) +
    lag(d_log_ratio_tot_beds, 2:10) +
    lag(d_log_ratio_tot_nights, 2:10) +
    lag(d_log_ratio_str_houses, 2:10) +
    lag(d_log_median_income, 2:10) +
    lag(d_log_population, 2:10) +
    lag(d_log_prov_immigration, 2:10) +
    lag(d_log_nominal_gdp, 2:10) +
    lag(d_log_interest_rate, 2:10),
  data = pdata1_cap,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("mun_istat", "year")
)

summary(sys_model_growth_cap_no, robust = TRUE, time.dummies = FALSE)
