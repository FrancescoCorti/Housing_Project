library(plm)
library(lmtest)
library(dplyr)
library(ivreg)
library(modelsummary)
library(tibble)
library(car)

# PROVINCE
df <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model.csv')

df_cap <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model_capital.csv')

pdata <- pdata.frame(df, index=c("prov","year"))

pdata_cap <- pdata.frame(df_cap, index=c("mun_istat","year"))



# VIF
fe_model <- plm(
  log_prov_buy_max_wgt ~
    #lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_vdw +
    log_prov_median_income_wgt +
    #log_prov_population +
    prov_density +
    log_reg_age_avg +
    log_over65 +
    log_prov_immigration +
    log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate +
    log_life +
    covid,
    data = pdata,
  index = c("prov", "year"),
  model = "pooling"
  )

vif(fe_model)


# PROVINCE FE 
fe_model <- plm(
  log_prov_buy_max_wgt ~
    #lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_vdw +
    log_prov_median_income_wgt +
    log_prov_population +
    #prov_density +
    log_reg_age_avg +
    log_over65 +
    log_prov_immigration +
    log_prov_unemployment +
    log_life +
    log_nominal_gdp +
    log_interest_rate +
    covid,
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
    #prov_density +
    log_reg_age_avg +
    log_over65 +
    log_prov_immigration +
    #log_prov_unemployment +
    log_life +
    log_nominal_gdp +
    log_interest_rate +
    covid,
    data = pdata_cap,
  index = c("mun_istat", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model_cap)



##########################################################################################################à

# PROVINCE - system-GMM (LEVEL)
sys_model <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_pca +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    log_prov_median_income_wgt +
    log_prov_population +
    #prov_density +
    log_reg_age_avg +
    #log_over65 +
    log_prov_immigration +
    #log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_pca, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    #lag(log_prov_unemployment, 2:10) +
    lag(log_prov_population, 2:10) +
    #lag(prov_density, 2:10) +
    lag(log_reg_age_avg, 2:10) +
    #lag(log_over65, 2:10) +
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
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    log_median_income +
    log_population +
    #prov_density +
    log_reg_age_avg +
    #log_over65 +
    log_prov_immigration +
    #log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_buy_max, 2:10) +
    lag(log_tourism_index_pca, 2:10) +
    lag(log_median_income, 2:10) +
    #lag(log_prov_unemployment, 2:10) +
    lag(log_population, 2:10) +
    #lag(prov_density, 2:10) +
    lag(log_reg_age_avg, 2:10) +
    #lag(log_over65, 2:10) +
    lag(log_prov_immigration, 2:10) +
    lag(log_nominal_gdp, 2:10) +
    lag(log_interest_rate, 2:10),
  data = pdata_cap,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_cap, robust = TRUE, time.dummies = FALSE)


# Compute quartiles
q <- quantile(pdata$prov_population, probs = c(0.25, 0.5, 0.75))

# Create four dummies
pdata$pop_q1 <- ifelse(pdata$prov_population <= q[1], 1, 0)                  # 0–25%
pdata$pop_q2 <- ifelse(pdata$prov_population > q[1] & pdata$prov_population <= q[2], 1, 0)   # 25–50%
pdata$pop_q3 <- ifelse(pdata$prov_population > q[2] & pdata$prov_population <= q[3], 1, 0)   # 50–75%
pdata$pop_q4 <- ifelse(pdata$prov_population > q[3], 1, 0)      

pdata$tourism_q1 <- pdata$d_log_tourism_index_vdw * pdata$pop_q1
pdata$tourism_q2 <- pdata$d_log_tourism_index_vdw * pdata$pop_q2
pdata$tourism_q3 <- pdata$d_log_tourism_index_vdw * pdata$pop_q3
pdata$tourism_q4 <- pdata$d_log_tourism_index_vdw * pdata$pop_q4


# PROVINCE - system-GMM (LEVEL) - POPULATION QUARTILES
sys_model_pop <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    tourism_q2 +
    tourism_q3 +
    tourism_q4 +
    log_prov_median_income_wgt +
    log_prov_population +
    #log_over65 +
    log_reg_age_avg +
    log_prov_immigration +
    #log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(tourism_q2, 2:10) +
    lag(tourism_q3, 2:10) +
    lag(tourism_q4, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    #lag(log_prov_unemployment, 2:10) +
    lag(log_prov_population, 2:10) +
    lag(log_reg_age_avg, 2:10) +
    #lag(log_over65, 2:10) +
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

summary(sys_model_pop, robust = TRUE, time.dummies = FALSE)


# Compute quartiles
q_buy <- quantile(pdata1$prov_buy_max_wgt, probs = c(0.25, 0.5, 0.75))

# Create four dummies
pdata$buy_q1 <- ifelse(pdata$prov_buy_max_wgt <= q_buy[1], 1, 0)
pdata$buy_q2 <- ifelse(pdata$prov_buy_max_wgt > q_buy[1] & pdata$prov_buy_max_wgt <= q_buy[2], 1, 0)
pdata$buy_q3 <- ifelse(pdata$prov_buy_max_wgt > q_buy[2] & pdata$prov_buy_max_wgt <= q_buy[3], 1, 0)
pdata$buy_q4 <- ifelse(pdata$prov_buy_max_wgt > q_buy[3], 1, 0)

# Interact with tourism growth
pdata$tourism_buy_q1 <- pdata$d_log_tourism_index_vdw * pdata$buy_q1
pdata$tourism_buy_q2 <- pdata$d_log_tourism_index_vdw * pdata$buy_q2
pdata$tourism_buy_q3 <- pdata$d_log_tourism_index_vdw * pdata$buy_q3
pdata$tourism_buy_q4 <- pdata$d_log_tourism_index_vdw * pdata$buy_q4


# PROVINCE - system-GMM (LEVEL) - POPULATION QUARTILES
sys_model_buy <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    tourism_buy_q2 +
    tourism_buy_q3 +
    tourism_buy_q4 +
    log_prov_median_income_wgt +
    log_prov_population +
    #log_over65 +
    log_reg_age_avg +
    log_prov_immigration +
    #log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(tourism_buy_q2, 2:10) +
    lag(tourism_buy_q3, 2:10) +
    lag(tourism_buy_q4, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    #lag(log_prov_unemployment, 2:10) +
    lag(log_prov_population, 2:10) +
    #lag(log_over65, 2:10) +
    lag(log_reg_age_avg, 2:10) +
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

summary(sys_model_buy, robust = TRUE, time.dummies = FALSE)


##################################################################################################


# PROVINCE GROWTH RATES
growth_data <- na.omit(pdata)

pdata1 <- pdata.frame(growth_data, index=c("prov","year"))

# PROVINCE - system-GMM (GROWTH)
sys_model_growth <- pgmm(
  d_log_prov_buy_max_wgt ~ 
    lag(d_log_prov_buy_max_wgt, 1) +
    d_log_tourism_index_vdw +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    d_log_prov_median_income_wgt +
    d_log_prov_population +
    #d_log_over65 +
    d_log_reg_age_avg +
    d_log_prov_immigration +
    #d_log_prov_unemployment +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_prov_buy_max_wgt, 2:10) +
    lag(d_log_tourism_index_vdw, 2:10) +
    lag(d_log_prov_median_income_wgt, 2:10) +
    #lag(d_log_prov_unemployment, 2:10) +
    lag(d_log_prov_population, 2:10) +
    #lag(d_log_over65, 2:10) +
    lag(d_log_reg_age_avg, 2:10) +
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


# CAPITALS GROWTH RATES
growth_data_cap <- na.omit(pdata_cap)

pdata1_cap <- pdata.frame(growth_data_cap, index=c("mun_istat","year"))

# CAPITALS - system-GMM (GROWTH)
sys_model_growth_cap <- pgmm(
  d_log_buy_max ~ 
    lag(d_log_buy_max, 1) +
    d_log_tourism_index_pca +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    d_log_median_income +
    d_log_population +
    #d_log_over65 +
    d_log_reg_age_avg +
    d_log_prov_immigration +
    #d_log_prov_unemployment +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_buy_max, 2:10) +
    lag(d_log_tourism_index_pca, 2:10) +
    lag(d_log_median_income, 2:10) +
    #lag(d_log_prov_unemployment, 2:10) +
    lag(d_log_population, 2:10) +
    #lag(d_log_over65, 2:10) +
    lag(d_log_reg_age_avg, 2:10) +
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


# INTERACTIONS
pdata1$tourism_population <- pdata1$d_log_tourism_index_pca * pdata1$log_prov_population
pdata1$tourism_density <- pdata1$d_log_tourism_index_vdw * pdata1$prov_density


sys_model_growth_i <- pgmm(
  d_log_prov_buy_max_wgt ~ 
    lag(d_log_prov_buy_max_wgt, 1) +
    d_log_tourism_index_pca +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    tourism_population +
    d_log_prov_median_income_wgt +
    d_log_prov_population +
    #d_log_over65 +
    d_log_reg_age_avg +
    d_log_prov_immigration +
    d_log_prov_unemployment +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_prov_buy_max_wgt, 2:10) +
    lag(d_log_tourism_index_pca, 2:10) +
    lag(d_log_prov_median_income_wgt, 2:10) +
    lag(tourism_population, 2:10) +
    #lag(tourism_income, 2:10) +
    #lag(d_log_prov_unemployment, 2:10) +
    lag(d_log_prov_population, 2:10) +
    #lag(d_log_over65, 2:10) +
    lag(d_log_reg_age_avg, 2:10) +
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

summary(sys_model_growth_i, robust = TRUE, time.dummies = FALSE)


gmm_tourism_quartile <- pgmm(
  d_log_prov_buy_max_wgt ~ 
    lag(d_log_prov_buy_max_wgt, 1) +
    tourism_q2 +
    tourism_q3 +
    tourism_q4 +
    d_log_prov_median_income_wgt +
    d_log_prov_population +
    #d_log_over65 +
    d_log_reg_age_avg +
    d_log_prov_immigration +
    #d_log_prov_unemployment +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_prov_buy_max_wgt, 2:10) +
    lag(tourism_q2, 2:10) +
    lag(tourism_q3, 2:10) +
    lag(tourism_q4, 2:10) +
    lag(d_log_prov_median_income_wgt, 2:10) +
    lag(d_log_prov_population, 2:10) +
    #lag(d_log_over65, 2:10) +
    lag(d_log_reg_age_avg, 2:10) +
    lag(d_log_prov_immigration, 2:10) +
    #lag(d_log_prov_unemployment, 2:10) +
    lag(d_log_nominal_gdp, 2:10) +
    lag(d_log_interest_rate, 2:10),
  data = pdata1,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld",
  index = c("prov", "year")
)

summary(gmm_tourism_quartile, robust = TRUE, time.dummies = FALSE)



gmm_tourism_buy <- pgmm(
  d_log_prov_buy_max_wgt ~ 
    lag(d_log_prov_buy_max_wgt, 1) +
    tourism_buy_q2 +
    tourism_buy_q3 +
    tourism_buy_q4 +
    d_log_prov_median_income_wgt +
    tourism_population +
    d_log_prov_population +
    #d_log_over65 +
    d_log_reg_age_avg +
    d_log_prov_immigration +
    d_log_prov_unemployment +
    d_log_nominal_gdp +
    d_log_interest_rate |
    lag(d_log_prov_buy_max_wgt, 2:10) +
    lag(tourism_buy_q2, 2:10) +
    lag(tourism_buy_q3, 2:10) +
    lag(tourism_buy_q4, 2:10) +
    lag(d_log_prov_median_income_wgt, 2:10) +
    lag(d_log_prov_population, 2:10) +
    #lag(d_log_over65, 2:10) +
    lag(d_log_reg_age_avg) +
    lag(d_log_prov_immigration, 2:10) +
    #lag(d_log_prov_unemployment, 2:10) +
    lag(d_log_nominal_gdp, 2:10) +
    lag(d_log_interest_rate, 2:10),
  data = pdata1,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld",
  index = c("prov", "year")
)

summary(gmm_tourism_buy, robust = TRUE, time.dummies = FALSE)

