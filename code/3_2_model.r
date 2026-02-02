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
library(SDPDmod)


############################################

# LOAD THE DATA

# PROVINCE
df <- read.csv('C:/Users/HP/Desktop/Traineeship/Code/code/datasets/model.csv')

pdata <- pdata.frame(df, index=c("prov","year"))


############################################


# CONTROL TESTS

# VIF - MAIN MODEL
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
moran.test(df_2019$log_prov_buy_avg_wgt_de,
           listw = W_q,
           zero.policy = TRUE)


sp_2019 <- pdata %>%
  filter(year == 2019) %>%
  arrange(prov)   # MUST match prov_sf order

nrow(sp_2019)
length(W_q$neighbours)

##########################################################################

# SAM

# Null model
sdm_null <- lagsarlm(
  log_prov_buy_avg_wgt_de ~ 1,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

# SAM - z
sar_model_z <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_z +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_reg_age_avg,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_model_z)

impacts_sar_z <- impacts(
  sar_model_z,
  listw = W_q,
  R = 1000
)

summary(impacts_sar_z, zstats = TRUE)

R2_LR <- 1 - exp(
  (2 / nrow(sp_2019)) *
  (logLik(sdm_null) - logLik(sar_model_z))
)
R2_LR


# SAM - no
sar_model_no <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_reg_age_avg,
  data = sp_2019,
  listw = W_q,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sar_model_no)

impacts_sar_no <- impacts(
  sar_model_no,
  listw = W_q,
  R = 1000
)

summary(impacts_sar_no, zstats = TRUE)

R2_LR <- 1 - exp(
  (2 / nrow(sp_2019)) *
  (logLik(sdm_null) - logLik(sar_model_no))
)
R2_LR

############################################################

# SDM

# SDM - z
sdm_z <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_z +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_reg_age_avg,
  data = sp_2019,
  listw = W_q,
  Durbin = TRUE,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sdm_z)

impacts_sdm_z <- impacts(
  sdm_z,
  listw = W_q,
  R = 1000
)

summary(impacts_sdm_z, zstats = TRUE)

R2_LR <- 1 - exp(
  (2 / nrow(sp_2019)) *
  (logLik(sdm_null) - logLik(sdm_z))
)
R2_LR


# SDM - no
sdm_no <- lagsarlm(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = sp_2019,
  listw = W_q,
  Durbin = TRUE,
  zero.policy = TRUE,
  method = "eigen"
)

summary(sdm_no)


impacts_sdm_no <- impacts(
  sdm_no,
  listw = W_q,
  R = 1000
)

summary(impacts_sdm_no, zstats = TRUE)

R2_LR <- 1 - exp(
  (2 / nrow(sp_2019)) *
  (logLik(sdm_null) - logLik(sdm_no))
)
R2_LR

#############################################################

W_mat <- listw2mat(W_q)


mod1<-SDPDm(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment, 
  data = pdata, 
  W = W_mat,
  index = c("prov",'year'),
  model = "sdm", 
  effect = "twoways",
  dynamic = F,
  tlaginfo = list(ind = NULL, tl = F, stl = F)
)
  
summary(mod1)





##############################################################

# SAM - z - fe
sar_z_fe <- spml(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_z +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment +
    log_reg_age_avg,
  data = pdata,
  listw = W_q,
  model = "within",
  lag = TRUE,
  effect = "twoways"
)

summary(sar_z_fe)


y <- sar_z_fe$model[, 1]
y_hat <- sar_z_fe$fitted.values

R2_within <- cor(y, y_hat)^2
R2_within


# SAR - rnk - fe
sar_rnk_fe <- spml(
  log_prov_buy_avg_wgt_de ~
    log_tourism_index_rnk +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = pdata,
  listw = W_q,
  model = "within",
  lag = TRUE,
  effect = "twoways"
)

summary(sar_rnk_fe)


y <- sar_rnk_fe$model[, 1]
y_hat <- sar_rnk_fe$fitted.values

R2_within <- cor(y, y_hat)^2
R2_within


# SAR - no - fe
sar_no_fe <- spml(
  log_prov_buy_avg_wgt_de ~
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    log_prov_population +
    log_prov_median_income_wgt +
    log_prov_unemployment,
  data = pdata,
  listw = W_q,
  model = "within",
  lag = TRUE,
  effect = "twoways"
)

summary(sar_no_fe)

y <- sar_no_fe$model[, 1]
y_hat <- sar_no_fe$fitted.values

R2_within <- cor(y, y_hat)^2
R2_within

################################################################################

# spatial weights (within year)

all(levels(pdata$prov) == prov_sf$prov)

spatial_var <- c(
  "log_prov_buy_avg_wgt_de",
  "log_tourism_index_z",
  "log_prov_ratio_tot_nights",
  "log_prov_ratio_str_houses"
)

for (v in spatial_var) {
  pdata <- pdata %>%
    group_by(year) %>%
    mutate(
      !!paste0("W_", v) := lag.listw(
        W_q,
        .data[[v]],
        zero.policy = TRUE
      )
    ) %>%
    ungroup()
}






   
# PROVINCE - system-GMM - z - avg
sys_model_z_avg_de <- pgmm(
  log_prov_buy_avg_wgt_de ~ 
    lag(log_prov_buy_avg_wgt_de, 1) +
    W_log_prov_buy +
    log_tourism_index_z +
    W_log_tourism +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt_de, 2:10) +
    lag(W_log_prov_buy, 2:5) +
    lag(log_tourism_index_z, 2:10) +
    lag(W_log_tourism, 2:5) +
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


# PROVINCE - system-GMM - z - avg
sys_model_z_avg_de <- pgmm(
  log_prov_buy_avg_wgt_de ~ 
    lag(log_prov_buy_avg_wgt_de, 1) +
    W_log_prov_buy_avg_wgt_de +
    log_prov_ratio_tot_nights +
    log_prov_ratio_str_houses +
    W_log_prov_ratio_tot_nights +
    W_log_prov_ratio_str_houses +
    log_prov_population +
    log_reg_age_avg +
    log_prov_immigration +
    log_prov_median_income_wgt +
    log_prov_unemployment |
    lag(log_prov_buy_avg_wgt_de, 2:5) +
    lag(W_log_prov_buy_avg_wgt_de, 2:5) +
    lag(log_prov_ratio_tot_nights, 2:5) +
    lag(log_prov_ratio_str_houses, 2:5) +
    lag(W_log_prov_ratio_tot_nights, 2:5) +
    lag(W_log_prov_ratio_str_houses, 2:5) +
    lag(log_prov_ratio_str_houses, 2:5) +
    lag(log_prov_population, 2:5) +
    lag(log_prov_immigration, 2:5) +
    lag(log_prov_median_income_wgt, 2:5) +
    lag(log_prov_unemployment, 2:5),
  data = pdata,
  effect = "twoways",
  model = "twostep",
  collapse = TRUE,
  transformation = "ld", # ld for system-GMM and d for difference-GMM
  index = c("prov", "year")
)

summary(sys_model_z_avg_de, robust = TRUE, time.dummies = FALSE)
