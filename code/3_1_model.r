library(plm)
library(lmtest)
library(dplyr)
library(ivreg)
library(modelsummary)
library(tibble)

# PROVINCE
df <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model.csv')

pdata <- pdata.frame(df, index=c("prov","year"))

# PROVINCE FE 
fe_model <- plm(
  log_prov_buy_max_wgt ~
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_vdw +
    log_prov_median_income_wgt +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate,
    data = pdata,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model)


# PROVINCE - system-GMM (LEVEL)
sys_model <- pgmm(
  log_prov_buy_max_wgt ~ 
    lag(log_prov_buy_max_wgt, 1) +
    log_tourism_index_vdw +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    log_prov_median_income_wgt +
    log_prov_population +
    log_reg_age_avg +
    #over65 +
    log_prov_immigration +
    log_prov_unemployment +
    log_nominal_gdp +
    log_interest_rate |
    lag(log_prov_buy_max_wgt, 2:10) +
    lag(log_tourism_index_vdw, 2:10) +
    lag(log_prov_median_income_wgt, 2:10) +
    lag(log_prov_unemployment, 2:10) +
    lag(log_prov_population, 2:10) +
    #lag(reg_age_avg, 2:10) +
    #lag(over65, 2:10) +
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


# PROVINCE GROWTH RATES
df1 <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/datasets/prov_growth_model_data.csv')

pdata1 <- pdata.frame(df1, index=c("prov","year"))

# PROVINCE - system-GMM (GROWTH)
sys_model_growth <- pgmm(
  growth_log_buy_max ~ 
    lag(growth_log_buy_max, 1) +
    tourism_score +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    growth_log_median_income +
    growth_log_population +
    growth_log_reg_age_avg +
    #over65 +
    growth_log_prov_immigration +
    growth_unemployment_prov +
    growth_log_nominal_gdp |
    lag(growth_log_buy_max, 2:10) +
    lag(growth_log_median_income, 2:10) +
    lag(unemployment_prov, 2:10) +
    lag(growth_log_population, 2:10) +
    #lag(reg_age_avg, 2:10) +
    #lag(over65, 2:10) +
    lag(prov_net_movements, 2:10) +
    lag(growth_log_nominal_gdp, 2:10) +
    lag(tourism_score, 2:10),
  data = pdata1,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_growth, robust = TRUE, time.dummies = FALSE)



#COMPARE COEFFICIENTS
# System GMM
ar1  <- mtest(sys_model, order = 1)$p.value
ar2  <- mtest(sys_model, order = 2)$p.value
hansen <- sargan(sys_model)$p.value
ninst <- length(sys_model$W)

# System GMM (growth)
ar1_g  <- mtest(sys_model_growth, order = 1)$p.value
ar2_g  <- mtest(sys_model_growth, order = 2)$p.value
hansen_g <- sargan(sys_model_growth)$p.value
ninst_g <- length(sys_model_growth$W)


















# INTERACTION
pdata1$tourism_age <- pdata1$tourism_score * pdata1$reg_age_avg
pdata1$tourism_pop <- pdata1$tourism_score * pdata1$growth_log_population



# PROV SELECTED
df2 <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/datasets/prov_select_model_data.csv')

pdata2 <- pdata.frame(df2, index=c("prov","year"))

fe_model <- plm(
  log_buy_avg ~
    #lag(log_buy_avg, 1) +
    tourism_score +
    log_median_income +
    log_population +
    reg_age_avg +
    over65 +
    prov_net_movements +
    unemployment_prov +
    #dwellings : log_population_growth +
    #factor(type) : tourism_score,
    log_nominal_gdp,
    #log_tour_employed,
    data = pdata2,
  index = c("prov", "year"),
  model = "within",
  effect = "individual"
)

summary(fe_model)


# PROVINCE - system-GMM (LEVEL)
sys_model1 <- pgmm(
  log_buy_max ~ 
    lag(log_buy_max, 1) +
    tourism_score +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    log_median_income +
    log_population +
    reg_age_avg +
    #over65 +
    prov_net_movements +
    unemployment_prov +
    log_nominal_gdp |
    lag(log_buy_max, 2:10) +
    lag(log_median_income, 2:10) +
    lag(unemployment_prov, 2:10) +
    lag(log_population, 2:10) +
    #lag(reg_age_avg, 2:10) +
    #lag(over65, 2:10) +
    lag(prov_net_movements, 2:10) +
    lag(log_nominal_gdp, 2:10) +
    lag(tourism_score, 2:10),
  data = pdata2,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model1, robust = TRUE, time.dummies = FALSE)


# CITIES - system-GMM (GROWTH)
df3 <- read.csv('datasets/prov_select_growth_model_data.csv')

pdata3 <- pdata.frame(df3, index=c("prov","year"))


sys_model_growth1 <- pgmm(
  growth_log_buy_avg ~ 
    lag(growth_log_buy_avg, 1) +
    tourism_score +
    # ratio_hotel_beds +
    # ratio_hotel_mun +
    # ratio_str_beds +
    # ratio_str_mun +
    growth_log_median_income +
    growth_log_population +
    reg_age_avg +
    #over65 +
    prov_net_movements +
    unemployment_prov +
    growth_log_nominal_gdp |
    lag(growth_log_buy_avg, 2:10) +
    lag(growth_log_median_income, 2:5) +
    lag(unemployment_prov, 2:5) +
    lag(growth_log_population, 2:5) +
    #lag(reg_age_avg, 2:10) +
    #lag(over65, 2:10) +
    lag(prov_net_movements, 2:5) +
    lag(growth_log_nominal_gdp, 2:5) +
    lag(tourism_score, 2:5),
  data = pdata3,
  effect = "individual",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_growth1, robust = TRUE, time.dummies = FALSE)
