library(car)
library(caret)
library(splines)
library(ordinal)
library(nnet)
library(irr)

# ----------------------------------
# 1. Data Preparation
# ----------------------------------

data <- read.csv("C:/Users/Saif Shaikh/Downloads/Datasets/Zomato.csv")

colSums(is.na(data))

set.seed(123)
train_index <- createDataPartition(
  y = data$food_rating,   
  p = 0.8,       
  list = FALSE
)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

boxplot(train_data$price_for_two)
boxplot(train_data$delivery_time_min)
boxplot(train_data$distance_km)
boxplot(train_data$food_prep_time_min)
boxplot(train_data$restaurant_age_years)
boxplot(train_data$avg_monthly_orders)
boxplot(train_data$discount_percent)
boxplot(train_data$delivery_fee)
boxplot(train_data$late_delivery_rate)

# ----------------------------------
# 2. Feature Scaling
# ----------------------------------

num_vars <- c("price_for_two", "delivery_time_min", "distance_km",
              "food_prep_time_min", "restaurant_age_years",
              "avg_monthly_orders", "discount_percent",
              "delivery_fee", "late_delivery_rate", "packaging_rating",
              "hygiene_rating", "customer_support_rating", "cuisine_count")


train_means <- sapply(train_data[, num_vars], mean)
train_sds   <- sapply(train_data[, num_vars], sd)

train_data[, num_vars] <- scale(train_data[, num_vars],
                                center = train_means,
                                scale  = train_sds)

train_data$food_rating<- factor(train_data$food_rating,
                                levels = c(1, 2, 3, 4, 5),
                                ordered = TRUE)
train_data$city_tier<- factor(train_data$city_tier,
                              levels = c(1, 2, 3),
                              ordered = TRUE)
train_data$restaurant_type<-factor(train_data$restaurant_type)
train_data$veg_option<-factor(train_data$veg_option)
train_data$online_order<-factor(train_data$online_order)
train_data$table_booking<-factor(train_data$table_booking)


test_data[, num_vars] <- scale(test_data[, num_vars],
                               center = train_means,
                               scale  = train_sds)

test_data$food_rating <- factor(test_data$food_rating,
                                levels = c(1, 2, 3, 4, 5),
                                ordered = TRUE)
test_data$city_tier <- factor(test_data$city_tier,
                              levels = c(1, 2, 3),
                              ordered = TRUE)
test_data$restaurant_type <- factor(test_data$restaurant_type)
test_data$veg_option <- factor(test_data$veg_option)
test_data$online_order <- factor(test_data$online_order)
test_data$table_booking <- factor(test_data$table_booking)

# ----------------------------------
# 3. Multicollinearity Check
# ----------------------------------

vif_model <- lm(
  as.numeric(food_rating) ~ price_for_two + delivery_time_min + 
    distance_km + food_prep_time_min + restaurant_age_years +
    avg_monthly_orders + discount_percent + delivery_fee +
    late_delivery_rate + packaging_rating + hygiene_rating + 
    customer_support_rating + cuisine_count + city_tier +
    restaurant_type + veg_option + online_order + table_booking,
  data = train_data
)

vif(vif_model)

# ----------------------------------
# 4. Linearity with Log Odds Check
# ----------------------------------

model0<-clm(food_rating ~ 1,
            data = train_data)

model1 <- clm(
  food_rating ~ price_for_two + delivery_time_min + 
    distance_km + food_prep_time_min + restaurant_age_years +
    avg_monthly_orders + discount_percent + delivery_fee +
    late_delivery_rate + packaging_rating + hygiene_rating +
    customer_support_rating + cuisine_count,
  data = train_data
)

anova(model0,model1)

model2 <- clm(
  food_rating ~ price_for_two + delivery_time_min +
    distance_km + food_prep_time_min + restaurant_age_years +
    avg_monthly_orders + discount_percent + delivery_fee +
    late_delivery_rate + packaging_rating + hygiene_rating +
    customer_support_rating + ns(cuisine_count,4),
  data = train_data
)

anova(model1, model2)

model3 <- clm(
  food_rating ~ ns(price_for_two,4) + ns(delivery_time_min,4) +
    distance_km + food_prep_time_min + restaurant_age_years +
    avg_monthly_orders + discount_percent + delivery_fee +
    ns(late_delivery_rate,4) + packaging_rating + hygiene_rating +
    customer_support_rating + cuisine_count + city_tier +
    restaurant_type + veg_option + online_order + table_booking,
  data = train_data
)

# ----------------------------------
# 5. Proportional Odds Check
# ----------------------------------

nominal_test(model3)

model4 <- clm(
  food_rating ~ ns(price_for_two,4) + ns(delivery_time_min,4) +
    distance_km + ns(food_prep_time_min,4) + restaurant_age_years +
    avg_monthly_orders + discount_percent + delivery_fee +
    ns(late_delivery_rate,4) + packaging_rating + hygiene_rating +
    customer_support_rating + cuisine_count + city_tier +
    restaurant_type + veg_option + online_order + table_booking,
  nominal = ~ distance_km + discount_percent,
  data = train_data
)

summary(model4)

step_model<-step(model0,scope = formula(model4), direction = 'both')

# ----------------------------------
# 6. Ordinal Logistic Regression
# ----------------------------------

model5 <- clm(
  food_rating ~ packaging_rating + hygiene_rating + customer_support_rating + 
    ns(delivery_time_min, 3) + ns(food_prep_time_min, 3) + restaurant_type + 
    ns(price_for_two, 3) + city_tier + cuisine_count + online_order + 
    distance_km + veg_option + avg_monthly_orders + ns(late_delivery_rate, 3) +
    restaurant_age_years + discount_percent,
  nominal = ~ distance_km + discount_percent,
  data = train_data
)

summary(model5)
anova(model4,model5)
anova(model0,model5)
AIC(model5)
logLik(model5)
exp(coef(model5))

ord_pred_class <- predict(model5, newdata = test_data, type = 'class')
ord_pred_class <- factor(ord_pred_class$fit,
                         levels = c(1, 2, 3, 4, 5),
                         ordered = TRUE)

confusionMatrix(ord_pred_class,test_data$food_rating)

pred_num <- as.numeric(as.character(ord_pred_class))
true_num <- as.numeric(as.character(test_data$food_rating))

within1_acc <- mean(abs(pred_num - true_num) <= 1)
within1_acc

McFadden_Pseudo_R2 <- 1 - (as.numeric(logLik(model5)) / 
                             as.numeric(logLik(model0)))
McFadden_Pseudo_R2

# ----------------------------------
# 7. Multinomial Logistic Regression
# ----------------------------------

model6 <- multinom(
  food_rating ~ ns(price_for_two,3) + ns(delivery_time_min,3) +
    distance_km + ns(food_prep_time_min,3) + restaurant_age_years +
    avg_monthly_orders + discount_percent + ns(late_delivery_rate,3) + 
    packaging_rating + hygiene_rating + customer_support_rating + 
    cuisine_count + city_tier + restaurant_type + veg_option + 
    online_order,
  data = train_data
)

summary(model6)
AIC(model6)
logLik(model6)

coefs <- summary(model6)$coefficients
std_err <- summary(model6)$standard.errors
z_values <- coefs / std_err
p_values <- 2 * (1 - pnorm(abs(z_values)))
p_values
rbind(coefs,'p value'=p_values)

# ----------------------------------
# 8. Model Comparison
# ----------------------------------

AIC(model5, model6)
BIC(model5, model6)

length(coef(model5))
length(coef(model6))

ord_pred_class <- predict(model5, newdata = test_data, type = 'class')
ord_pred_class <- factor(ord_pred_class$fit,
                         levels = c(1, 2, 3, 4, 5),
                         ordered = TRUE)
ord_pred_prob <- predict(model5, newdata = test_data, type = "prob")

nom_pred_class <- predict(model6, test_data, type = "class")
nom_pred_class <- factor(nom_pred_class,
                         levels = c(1, 2, 3, 4, 5),
                         ordered = TRUE)
nom_pred_prob  <- predict(model6, test_data, type = "prob")

mean(ord_pred_class == test_data$food_rating)
mean(nom_pred_class == test_data$food_rating)

ord_mae <- mean(abs(
  as.numeric(ord_pred_class) - as.numeric(test_data$food_rating)
))

nom_mae <- mean(abs(
  as.numeric(nom_pred_class) - as.numeric(test_data$food_rating)
))

ord_mae
nom_mae

kappa2(data.frame(ord_pred_class, test_data$food_rating),
       weight = "squared")

kappa2(data.frame(nom_pred_class, test_data$food_rating),
       weight = "squared")

confusionMatrix(nom_pred_class,test_data$food_rating)

# Ordinal model preferred due to:
# - Lower AIC
# - Comparable or better MAE
# - Fewer parameters (parsimony)
# - Respect for ordered structure


# ----------------------------------
# 9. Key Insights
# ----------------------------------

model_metrics <- data.frame(
  Exact_Accuracy = mean(ord_pred_class == test_data$food_rating),
  Within1_Accuracy = within1_acc,
  MAE = ord_mae,
  Kappa = kappa2(data.frame(ord_pred_class, test_data$food_rating), weight = "squared"),
  McFadden_R2 = as.numeric(McFadden_Pseudo_R2)
)

model_metrics

cat("Key Business Insights:\n
    - Higher packaging and hygiene ratings significantly increase probability of 4-5 star ratings.\n
    - Increase in late delivery rate reduces likelihood of high ratings.\n
    - Delivery time and price shows non-linear effect on customer satisfaction.\n
    - Ordinal model performs better than multinomial due to ordered nature of ratings.")
