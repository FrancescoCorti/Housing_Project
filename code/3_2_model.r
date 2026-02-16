library(plm)
library(lmtest)
library(dplyr)
library(modelsummary)
library(tibble)
library(car)

# spatial models
library(sf)
library(spdep)
library(spatialreg)
library(splm)

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

# VIF - z
vif <- plm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_z +
    log_prov_median_income_wgt +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate +
    covid_all,
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
    covid_all,
    data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(vif)


##########################################################

# OLS FIXED-EFFECT

# PROVINCE FE (no index) - avg
fe_model_no_avg <- plm(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment,
    #log_real_gdp +
    #log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "twoways"
)

summary(fe_model_no_avg)

# PROVINCE FE (z) - avg
fe_model_z_avg <- plm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment,
    #log_real_gdp +
    #log_interest_rate,
  data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "twoways"
)

summary(fe_model_z_avg)

##############################################################################

# OLS FIXED-EFFECT - NATIONAL CONTROLS

# PROVINCE FE (no index) - avg - nat
fe_model_no_avg_nat <- plm(
  log_prov_buy_avg_wgt ~
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

summary(fe_model_no_avg_nat)

# PROVINCE FE (z) - avg - nat
fe_model_z_avg_nat <- plm(
  log_prov_buy_avg_wgt ~
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

summary(fe_model_z_avg_nat)

#######################################################################

# SYSTEM-GMM

# system-GMM - no - avg
sys_no_avg <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_no_avg, robust = TRUE, time.dummies = FALSE)

# system-GMM - z - avg
sys_z_avg <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_z_avg, robust = TRUE, time.dummies = FALSE)

##################################################################################################

# SYSTEM-GMM - NATIONAL CONTROLS

# system-GMM - no - avg - nat
sys_no_avg_nat <- pgmm(
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
    log_interest_rate |
    lag(log_prov_buy_avg_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 1:10) +
    lag(log_interest_rate, 1:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_no_avg_nat, robust = TRUE, time.dummies = FALSE)

# system-GMM - z - avg - nat
sys_z_avg_nat <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_avg_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_real_gdp, 1:10) +
    lag(log_interest_rate, 1:10),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_z_avg_nat, robust = TRUE, time.dummies = FALSE)

##################################################################################################

# SYSTEM-GMM - minimum

# system-GMM - no - min
sys_no_min <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_no_min, robust = TRUE, time.dummies = FALSE)

# system-GMM - z - min
sys_z_min <- pgmm(
  log_prov_buy_min_wgt ~ 
    lag(log_prov_buy_min_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_min_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_z_min, robust = TRUE, time.dummies = FALSE)

##################################################################################################

# SYSTEM-GMM - maximum

# system-GMM - no - max
sys_no_max <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_prov_ratio_tot_nights, 2:10) +
    lag(log_prov_ratio_str_houses, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_no_max, robust = TRUE, time.dummies = FALSE)

# system-GMM - z - max
sys_z_max <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_z, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_prov_median_income_wgt, 1:10) +
    lag(log_prov_unemployment, 2:10),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_z_max, robust = TRUE, time.dummies = FALSE)

##################################################################################################

# SYSTEM-GMM - COVID

# system-GMM - no - avg - covid
sys_no_avg_cov <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    covid_all+ 
    covid_all:log_prov_ratio_tot_nights +
    covid_all:log_prov_ratio_str_houses +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_prov_ratio_tot_nights, 2:7) +
    lag(log_prov_ratio_str_houses, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 1:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(covid_all:log_prov_ratio_tot_nights, 2:7) +
    lag(covid_all:log_prov_ratio_str_houses, 2:7) +
    lag(log_real_gdp, 1:7) +
    lag(log_interest_rate, 1:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_no_avg_cov, robust = TRUE, time.dummies = FALSE)


# system-GMM - z - avg - cov
sys_z_avg_cov <- pgmm(
  log_prov_buy_avg_wgt ~ 
    lag(log_prov_buy_avg_wgt, 1) +
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    covid_all +
    covid_all:log_tourism_index_z +
    log_real_gdp +
    log_interest_rate |
    lag(log_prov_buy_avg_wgt, 2:7) +
    lag(log_tourism_index_z, 2:7) +
    lag(log_prov_population, 2:7) +
    lag(log_prov_immigration, 2:7) +
    lag(log_prov_median_income_wgt, 1:7) +
    lag(log_prov_unemployment, 2:7) +
    lag(covid_all:log_tourism_index_z, 2:7) +
    lag(log_real_gdp, 1:7) +
    lag(log_interest_rate, 1:7),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_z_avg_cov, robust = TRUE, time.dummies = FALSE)

#########################################################

# SYSTEM-GMM - COVID - 2020-2021 AND 2022

# system-GMM - no - avg - cov - 2
sys_no_avg_cov_2 <- pgmm(
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
    covid_20_21 +
    covid_2022 +
    covid_20_21:log_prov_ratio_tot_nights +
    covid_20_21:log_prov_ratio_str_houses +
    covid_2022:log_prov_ratio_tot_nights +
    covid_2022:log_prov_ratio_str_houses |
    lag(log_prov_buy_avg_wgt, 2:6) +
    lag(log_prov_ratio_tot_nights, 2:6) +
    lag(log_prov_ratio_str_houses, 2:6) +
    lag(log_prov_population, 2:6) +
    lag(log_prov_immigration, 2:6) +
    lag(log_prov_median_income_wgt, 1:6) +
    lag(log_prov_unemployment, 2:6) +
    lag(log_real_gdp, 1:6) +
    lag(log_interest_rate, 1:6) +
    lag(covid_20_21:log_prov_ratio_tot_nights, 2:6) +
    lag(covid_20_21:log_prov_ratio_str_houses, 2:6) +
    lag(covid_2022:log_prov_ratio_tot_nights, 2:6) +
    lag(covid_2022:log_prov_ratio_str_houses, 2:6),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_no_avg_cov_2, robust = TRUE, time.dummies = FALSE)

# system-GMM - z - avg - covid - 2
sys_z_avg_cov_2 <- pgmm(
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
    covid_20_21 +
    covid_2022 +
    covid_20_21:tourism_index_z +
    covid_2022:tourism_index_z |
    lag(log_prov_buy_avg_wgt, 2:6) +
    lag(log_tourism_index_z, 2:6) +
    lag(log_prov_population, 2:6) +
    lag(log_prov_immigration, 2:6) +
    lag(log_prov_median_income_wgt, 1:6) +
    lag(log_prov_unemployment, 2:6) +
    lag(log_real_gdp, 1:6) +
    lag(log_interest_rate, 1:6) +
    lag(covid_20_21:tourism_index_z, 2:6) +
    lag(covid_2022:tourism_index_z, 2:6),
  data = pdata,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_z_avg_cov_2, robust = TRUE, time.dummies = FALSE)

########################################################


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

# Moran I test - buy
moran.test(df_2019$log_prov_buy_avg_wgt,
           listw = W_q,
           zero.policy = TRUE)

# Moran I test - z
moran.test(df_2019$log_tourism_index_z,
           listw = W_q,
           zero.policy = TRUE)

# Moran I test - nights
moran.test(df_2019$log_prov_ratio_tot_nights,
           listw = W_q,
           zero.policy = TRUE)

# Moran I test - str
moran.test(df_2019$log_prov_ratio_str_houses,
           listw = W_q,
           zero.policy = TRUE)


sp_2019 <- pdata %>%
  filter(year == 2019) %>%
  arrange(prov)   # MUST match prov_sf order

nrow(sp_2019)
length(W_q$neighbours)

##########################################################################

# SPATIAL AUTOREGRESSIVE MODEL - STATIC

# SAR - no
sar_no <- lagsarlm(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_no)

# SAR - z
sar_z <- lagsarlm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_z)

############################################################

# SPATIAL DURBIN MODEL - STATIC 

# SDM - no
sdm_no <- lagsarlm(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  Durbin = TRUE,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sdm_no)

# SDM - z
sdm_z <- lagsarlm(
  log_prov_buy_avg_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  Durbin = TRUE,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sdm_z)

#############################################################

fe_ols <- plm(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  model = "within",
  effect = "individual"
)

# SPATIAL AUTOREGRESSIVE MODEL- FIXED EFFECTS

# SAR - no - fe
sar_no_fe <- spml(
  log_prov_buy_avg_wgt ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  listw = W_q,
  model = "within",
  lag = TRUE,
  effect = "individual",
  spatial.error = "none"
)

summary(sar_no_fe)

index <- index(pdata)       # returns a dataframe with columns: prov, year
id <- index[,1]             # first column = "prov"

y <- sar_no_fe$model[,1]
y_hat <- sar_no_fe$fitted.values

# get the individual IDs from pdata
id <- index(pdata)[,1]

# compute within-transformed variables
y_within <- y - ave(y, id, FUN = mean)
yhat_within <- y_hat - ave(y_hat, id, FUN = mean)

# pseudo R² for within
R2_within <- cor(y_within, yhat_within)^2
R2_within


# SAR - z - fe
sar_z_fe <- spml(
  log_prov_buy_avg_wgt ~
    log_tourism_index_z +
    log_prov_population +
    log_reg_age_avg +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_real_gdp +
    log_interest_rate,
  data = pdata,
  listw = W_q,
  model = "within",
  lag = TRUE,
  effect = "individual",
  spatial.error = "none"
)

summary(sar_z_fe)


y <- sar_z_fe$model[,1]
y_hat <- sar_z_fe$fitted.values

# get the individual IDs from pdata
id <- index(pdata)[,1]

# compute within-transformed variables
y_within <- y - ave(y, id, FUN = mean)
yhat_within <- y_hat - ave(y_hat, id, FUN = mean)

# pseudo R² for within
R2_within <- cor(y_within, yhat_within)^2
R2_within

