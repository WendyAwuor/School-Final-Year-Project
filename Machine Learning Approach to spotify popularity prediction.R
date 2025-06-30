############################################################################################
### COMBINED SPOTIFY POPULARITY ANALYSIS: MULTIPLE REGRESSION MODELS THEN CLASSIFICATION ###
############################################################################################

# Clear environment and set seed for reproducibility
# rm(list = ls())
set.seed(42)

# Load essential libraries
library(tidyverse)
library(caret)
library(corrplot)
library(skimr)
library(DataExplorer)
library(vip)
library(ranger)
library(xgboost)
library(e1071)
library(scales)
library(lubridate)
library(GGally)
library(grid)
library(recipes)
library(pROC)
library(gridExtra)
library(ggpubr)
library(knitr)
library(MLmetrics)

# Start PDF device for all outputs
pdf("combined_reg_classy_spotify_analysis_results_multiple_reg.pdf", width = 12, height = 10)

# =============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION (SHARED)
# =============================================================================

# Load the csv file
spotify_charts <- read_csv("~/school docs/universal_top_spotify_songs.new.csv")

# Initial data exploration (same as before)
# ... (rest of the initial exploration code) ...
cat("====================\n")
cat("Initial Data Exploration\n")
cat("====================\n")
glimpse(spotify_charts)
cat("\nSummary:\n")
print(summary(spotify_charts))
cat("\nSkim:\n")
print(skim(spotify_charts))

# Check for missing values
missing_values <- colSums(is.na(spotify_charts))
cat("\nMissing values per column:\n")
print(missing_values[missing_values > 0])

# Visualize missing data pattern
plot_missing_plot <- plot_missing(spotify_charts)
print(plot_missing_plot)

# =============================================================================
# 2. DATA CLEANING AND FEATURE ENGINEERING (SHARED)
# =============================================================================

# combining all the codes
spotify_charts <- spotify_charts %>%
  group_by(spotify_id) %>%
  mutate(
    market_count = n_distinct(country, na.rm = TRUE),
    other_charted_countries = paste(country[!duplicated(country)], collapse = ", ")
  ) %>%
  slice_max(order_by = popularity, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    artist_count = sapply(strsplit(artists, ","), length),
    snapshot_date = ymd(snapshot_date),
    album_release_date = ymd(album_release_date),
    days_out = as.numeric(snapshot_date - album_release_date),
    is_explicit = as.integer(is_explicit),
    duration_min = duration_ms / 60000
  ) %>%
  select(-duration_ms) # Remove the original duration_ms column

# Verify the column name change (optional)
colnames(spotify_charts)[colnames(spotify_charts) == "duration_min"] <- "duration_min"

view(spotify_charts)
cat("\n====================\n")
cat("Cleaned and Engineered Data (First few rows):\n")
cat("====================\n")
print(head(spotify_charts))

# Prepare dataset for regression modeling
regression_data <- spotify_charts %>%
  select(-country, -other_charted_countries, -snapshot_date, -name, -artists,
         -album_name, -album_release_date, -spotify_id) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), ~if_else(is.na(.), median(., na.rm = TRUE), .))) %>%
  filter(popularity != 0)
# arrange columns
regression_data <- regression_data %>%
  select(popularity, market_count, daily_rank, days_out, artist_count, daily_movement, weekly_movement,
         duration_min, is_explicit, mode, danceability, energy, loudness, speechiness,
         acousticness, instrumentalness, time_signature, liveness, valence, key, tempo)

cat("\n====================\n")
cat("Regression Modeling Data (Summary):\n")
cat("====================\n")
view(regression_data)
#=====================================================================
## DESCRIPTIVE STATISTICS, SCATTER AND CORRELATION PLOTS
#=====================================================================

# Function to compute basic statistics
spotify_stats <- function(column) {
  stats <- c(
    Mean = mean(column, na.rm = TRUE),
    Median = median(column, na.rm = TRUE),
    SD = sd(column, na.rm = TRUE),
    Variance = var(column, na.rm = TRUE),
    IQR = IQR(column, na.rm = TRUE)
  )
  return(stats)
}

# Loop through columns and compute statistics
stats_results <- lapply(regression_data, spotify_stats)
names(stats_results) <- colnames(regression_data)

# Convert the list of statistics to a data frame for better printing
stats_table <- as.data.frame(stats_results)
print(stats_table)

# Generate the four scatter plot matrices
plot1 <- ggpairs(regression_data, columns = 1:5)
plot2 <- ggpairs(regression_data, columns = 6:10)
plot3 <- ggpairs(regression_data, columns = 11:15)
plot4 <- ggpairs(regression_data, columns = 16:20)

# Create a layout matrix
layout_matrix <- matrix(c(1,2,3,4), nrow = 2, byrow = TRUE)

# Open a graphics device and use grid.newpage() to manually arrange plots
grid.newpage()
pushViewport(viewport(layout = grid.layout(nrow = 2, ncol = 2)))

# Print each ggmatrix object in its respective location
print(plot1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(plot2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
print(plot3, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
print(plot4, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))

# check popularity distribution
popularity_dist_plot <- ggplot(data=regression_data)+
  geom_bar(mapping=aes(x=popularity, fill = "skyblue"))+
  ggtitle('popularity dist. after cleaning')
print(popularity_dist_plot)

# Compute correlation matrix
correlation_matrix <- cor(regression_data, use = "complete.obs")
# Plot heatmap with correlation values
corrplot_plot <- corrplot(correlation_matrix,
                          method = "color",
                          tl.col = "black",
                          tl.srt = 45,
                          type = "upper",
                          addCoef.col = "black",
                          number.cex = 0.7,
                          number.digits = 3,
                          main = "Correlation Heatmap",
                          mar = c(0, 0, 2, 0))
print(corrplot_plot)

# =============================================================================
# 3. MULTIPLE REGRESSION MODEL TRAINING AND EVALUATION
# =============================================================================

cat("\n====================\n")
cat("Multiple Regression Model Training and Evaluation\n")
cat("====================\n")

# Split data for regression
set.seed(42) # Reset seed for consistent split
train_index_reg <- createDataPartition(regression_data$popularity, p = 0.8, list = FALSE)
train_data_reg <- regression_data[train_index_reg, ]
test_data_reg <- regression_data[-train_index_reg, ]

# Create preprocessing recipe for regression
preprocess_recipe_reg <- recipe(popularity ~ ., data = train_data_reg) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

prep_recipe_reg <- prep(preprocess_recipe_reg, training = train_data_reg)
train_processed_reg <- bake(prep_recipe_reg, new_data = train_data_reg)
test_processed_reg <- bake(prep_recipe_reg, new_data = test_data_reg)

# Define resampling strategy
ctrl_reg <- trainControl(method = "cv", number = 5, verboseIter = FALSE, savePredictions = "final")

# Initialize a list to store regression models
regression_models <- list()

# 1. Random Forest
cat("\nTraining Random Forest (Regression)...\n")
regression_models$R.F <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "ranger",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(mtry = c(5, 7, 9), splitrule = "variance", min.node.size = c(1, 3)),
  importance = 'impurity',
  ntree = 250
)
cat("Random Forest (Regression) trained.\n")
print(regression_models$R.F)

# 2. Gradient Boosting
cat("\nTraining Gradient Boosting (Regression)...\n")
regression_models$GBM <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "gbm",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(n.trees = c(150, 250), interaction.depth = c(3, 5), shrinkage = c(0.1, 1), n.minobsinnode = 10),
  verbose = FALSE
)
cat("Gradient Boosting (Regression) trained.\n")
print(regression_models$GBM)

# 3. XGBoost
cat("\nTraining XGBoost (Regression)...\n")
regression_models$XGB <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "xgbTree",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(nrounds = c(150, 250), max_depth = c(3, 6), eta = c(0.1, 1), gamma = 0,
                         colsample_bytree = c(0.5, 0.75), min_child_weight = c(1, 3), subsample = 0.75),
  verbose = FALSE
)
cat("XGBoost (Regression) trained.\n")
print(regression_models$XGB)

# Evaluate all regression models on the test set
regression_evaluation_results <- lapply(names(regression_models), function(model_name) {
  predictions <- predict(regression_models[[model_name]], newdata = test_processed_reg)
  performance <- postResample(pred = predictions, obs = test_processed_reg$popularity)
  cat(paste0("\nPerformance of ", model_name, " (Regression) on Test Set:\n"))
  print(performance)
  
  # Return both performance metrics and predictions as a list
  return(list(performance = performance, predictions = predictions))
})
names(regression_evaluation_results) <- names(regression_models)

# Compile performance metrics for comparison
regression_results_df <- data.frame(
  Model = names(regression_models),
  RMSE = sapply(regression_evaluation_results, function(x) x$performance["RMSE"]),
  Rsquared = sapply(regression_evaluation_results, function(x) x$performance["Rsquared"]),
  MAE = sapply(regression_evaluation_results, function(x) x$performance["MAE"])
)

# Transpose the dataframe for the desired output
regression_results_df <- as.data.frame(t(regression_results_df[,-1]))
colnames(regression_results_df) <- names(regression_models)
rownames(regression_results_df) <- c("RMSE", "Rsquared", "MAE")


print("\nRegression Model Performance Comparison:")
print(regression_results_df)

# Determine the best performing regression model based on Rsquared
best_reg_model_name <- names(which.max(regression_results_df["Rsquared",]))
best_reg_model <- regression_models[[best_reg_model_name]]

cat(paste("\nBest performing regression model (based on Rsquared):", best_reg_model_name, "\n"))

# Get predictions from the best regression model
predictions_best_reg <- predict(best_reg_model, newdata = test_processed_reg)

# ----------------------------------------------------------------------------------------------------------------------
# Visualization of Regression Model Performance
# ----------------------------------------------------------------------------------------------------------------------

# Create a data frame for plotting performance metrics
performance_plot_data <- gather(regression_results_df, key = "Model", value = "Value") %>%
  mutate(Metric = rep(rownames(regression_results_df), times = length(regression_models)))

# Plotting RMSE with numeric values
rmse_plot <- ggplot(performance_plot_data %>% filter(Metric == "RMSE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = -0.25), size = 3) + # Add numeric labels
  labs(title = "Comparison of RMSE for Regression Models", y = "RMSE") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(rmse_plot)

# Plotting Rsquared with numeric values
rsquared_plot <- ggplot(performance_plot_data %>% filter(Metric == "Rsquared"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = -0.25), size = 3) + # Add numeric labels
  labs(title = "Comparison of Rsquared for Regression Models", y = "R-squared") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(rsquared_plot)

# Plotting MAE with numeric values
mae_plot <- ggplot(performance_plot_data %>% filter(Metric == "MAE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = -0.25), size = 3) + # Add numeric labels
  labs(title = "Comparison of MAE for Regression Models", y = "MAE") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(mae_plot)

# Create a data frame for plotting predicted vs actual values
plot_data_reg <- data.frame(
  Model = rep(names(regression_models), each = nrow(test_processed_reg)),
  Predicted = unlist(lapply(regression_evaluation_results, function(x) x$predictions)),
  Actual = rep(test_processed_reg$popularity, times = length(regression_models))
)

# Scatter plot of predicted vs actual values for each model
regression_scatter_plot <- ggplot(plot_data_reg, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +
  geom_smooth(aes(group = Model), method = "lm", se = FALSE) +
  facet_wrap(~Model, scales = "free") +
  labs(title = "Predicted vs. Actual Popularity for Regression Models",
       x = "Actual Popularity",
       y = "Predicted Popularity") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(regression_scatter_plot)

# ----------------------------------------------------------------------------------------------------------------------
# Feature Importance Plot (Best Regression Model)
# ----------------------------------------------------------------------------------------------------------------------

# Plot feature importance for the best regression model
vip_plot_reg <- vip(best_reg_model,
                    main = paste("Feature Importance in", best_reg_model_name, "(Regression)"),
                    color = "skyblue")
print(vip_plot_reg)

# =============================================================================
# 4. PREPARE DATA FOR CLASSIFICATION (FROM BEST REGRESSION OUTPUT)
# =============================================================================

cat("\n====================\n")
cat("Classification Based on Best Regression Model Predictions\n")
cat("====================\n")

# Define popularity thresholds
very_high_threshold <- 75
high_threshold <- 50
very_low_threshold <- 25

# Categorize the predicted popularity from the best regression model
predicted_popularity_level <- case_when(
  predictions_best_reg >= very_high_threshold ~ "very_high",
  predictions_best_reg < very_low_threshold ~ "very_low",
  predictions_best_reg >= high_threshold ~ "high",
  predictions_best_reg < high_threshold ~ "low"
)
predicted_popularity_level <- factor(predicted_popularity_level, levels = c("very_low", "low", "high", "very_high"))

# Create the actual popularity levels from the test set
actual_popularity_level <- case_when(
  test_data_reg$popularity >= very_high_threshold ~ "very_high",
  test_data_reg$popularity < very_low_threshold ~ "very_low",
  test_data_reg$popularity >= high_threshold ~ "high",
  test_data_reg$popularity < high_threshold ~ "low"
)
actual_popularity_level <- factor(actual_popularity_level, levels = c("very_low", "low", "high", "very_high"))

# Prepare data for classification model training
classification_data <- data.frame(
  popularity_level = actual_popularity_level,  # Use the actual levels from the test set
  test_processed_reg
)
# Remove the original 'popularity' column to avoid redundancy/potential issues
classification_data <- classification_data %>% select(-popularity)

# Split data for classification
set.seed(42)
train_index_class <- createDataPartition(classification_data$popularity_level, p = 0.8, list = FALSE)
train_data_class <- classification_data[train_index_class, ]
test_data_class <- classification_data[-train_index_class, ]

# =============================================================================
# 5. CLASSIFICATION MODEL TRAINING AND EVALUATION
# =============================================================================
cat("\n====================\n")
cat("Classification Model Training and Evaluation\n")
cat("====================\n")

# Custom summary function for multi-class ROC
multiClassSummary <- function (data, lev = NULL, model = NULL) {
  if (length(lev) > 2) {
    rocs <- pROC::multiclass.roc(data$obs, data[, lev])
    auc <- pROC::auc(rocs)
    names(auc) <- "AUC"
    accuracy <- mean(data$obs == data$pred)
    names(accuracy) <- "Accuracy"
    return(c(AUC = auc, Accuracy = accuracy))
  } else {
    return(defaultSummary(data, lev, model))
  }
}

# Define resampling strategy for classification
trainControl_roc <- trainControl(method = "cv",
                                 number = 5,
                                 allowParallel = TRUE,
                                 summaryFunction = multiClassSummary,
                                 classProbs = TRUE,
                                 savePredictions = TRUE)

# Initialize a list to store classification models
classification_models <- list()

# Train a classification model (Random Forest, for example)
cat("\nTraining Classification Model (Random Forest)...\n")
classification_models$R.F <- train(
  popularity_level ~ .,
  data = train_data_class,
  method = "ranger",
  trControl = trainControl_roc,
  tuneGrid = expand.grid(mtry = c(5, 7, 9),
                         min.node.size = c(1, 3, 5),
                         splitrule = "gini"),
  num.trees = 250,
  metric = "AUC"
)
cat("Random Forest Classification trained.\n")
print(classification_models$R.F)
print(classification_models$R.F$results)

# Train XGBoost
cat("\nTraining Classification Model (XGBoost)...\n")
classification_models$XGB <- train(
  popularity_level ~ .,
  data = train_data_class,
  method = "xgbTree",
  trControl = trainControl_roc,
  tuneGrid = expand.grid(nrounds = c(150, 250),
                         max_depth = c(3, 6),
                         eta = c(0.1, 1),
                         gamma = 0,
                         colsample_bytree = c(0.5, 0.75),
                         min_child_weight = c(1, 3),
                         subsample = 0.75),
  metric = "AUC",
  verbose = FALSE
)
cat("XGBoost Classification trained.\n")
print(classification_models$XGB)
print(classification_models$XGB$results)

# =============================================================================
# 6. PLOT AUC/ROC CURVES (CLASSIFICATION)
# =============================================================================

cat("\n====================\n")
cat("Plotting AUC/ROC Curves (Classification)\n")
cat("====================\n")

# Function to plot ROC curves and return plot object
plot_roc_curves <- function(model, test_data, model_name) {
  # Get predicted probabilities on the test set
  pred_probs_class <- predict(model, newdata = test_data %>% select(-popularity_level), type = "prob")
  
  # Get class levels
  class_levels <- levels(test_data$popularity_level)
  roc_objects_class <- list()
  
  for (i in 1:length(class_levels)) {
    current_class <- class_levels[[i]]
    binary_response <- ifelse(test_data$popularity_level == current_class, 1, 0)
    predictor <- pred_probs_class[, current_class]
    roc_objects_class[[current_class]] <- roc(response = binary_response, predictor = predictor)
  }
  
  # Create the multi-class ROC plot with AUC values
  plot(roc_objects_class[[1]], col = 1, main = paste("One-vs-Rest ROC Curves (", model_name, ")", sep=""),
       xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate (Sensitivity)",
       print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.1 * (length(roc_objects_class) - 1) + 0.1, print.auc.x = 0.6)
  if (length(roc_objects_class) > 1) {
    for (i in 2:length(roc_objects_class)) {
      plot(roc_objects_class[[i]], add = TRUE, col = i,
           print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.1 * (length(roc_objects_class) - i) + 0.1, print.auc.x = 0.6)
    }
    # Add a legend
    legend("bottomright", legend = names(roc_objects_class), col = 1:length(roc_objects_class), lty = 1)
  }
  # Calculate and store AUC for each class
  auc_values_class <- sapply(roc_objects_class, auc)
  auc_table_class <- data.frame(Class = names(auc_values_class), AUC = auc_values_class)
  
  print(recordPlot()) # Print the ROC plot to the plots pane
  return(list(plot = recordPlot(), auc_table = auc_table_class))
}

# Plot ROC curves for Random Forest and XGBoost
rf_roc_results <- plot_roc_curves(classification_models$R.F, test_data_class, "Random Forest")
cat("\nAUC for each class (One-vs-Rest on Test Set - Random Forest):\n")
print(rf_roc_results$auc_table)

xgb_roc_results <- plot_roc_curves(classification_models$XGB, test_data_class, "XGBoost")
cat("\nAUC for each class (One-vs-Rest on Test Set - XGBoost):\n")
print(xgb_roc_results$auc_table)

# =============================================================================
# 7. CONFUSION MATRIX AND CLASSIFICATION METRICS
# =============================================================================

cat("\n====================\n")
cat("Confusion Matrix and Classification Metrics\n")
cat("====================\n")

# Function to evaluate and print classification metrics
evaluate_classification_model <- function(model, test_data, model_name) {
  # Make predictions on the test set
  predictions_class <- predict(model, newdata = test_data %>% select(-popularity_level))
  
  # Create confusion matrix
  conf_matrix_class <- confusionMatrix(data = predictions_class, reference = test_data$popularity_level)
  
  # Print confusion matrix
  cat(paste("\nConfusion Matrix (", model_name, " on Test Set):\n", sep = ""))
  print(conf_matrix_class)
  
  # Plot confusion matrix
  if (exists("draw_confusion_matrix")) {
    print(draw_confusion_matrix(conf_matrix_class, main = paste("Confusion Matrix - ", model_name, sep="")))
  } else if (requireNamespace("lattice", quietly = TRUE)) {
    print(lattice::levelplot(conf_matrix_class$table,
                             main = paste("Confusion Matrix - ", model_name, sep=""),
                             xlab = "Predicted Class",
                             ylab = "Actual Class",
                             col.regions = colorRampPalette(c("white", "lightblue"))(20),
                             scales = list(x = list(rot = 90))))
  } else {
    cat("Neither `draw_confusion_matrix` nor `lattice` package is available. Please install `lattice` for confusion matrix plotting.\n")
  }
  
  # Extract overall metrics
  overall_metrics_class <- data.frame(conf_matrix_class$overall)
  cat("\nOverall Classification Metrics:\n")
  print(overall_metrics_class)
  
  # Extract class-specific metrics
  class_metrics_class <- data.frame(conf_matrix_class$byClass)
  cat("\nClass-Specific Classification Metrics:\n")
  print(class_metrics_class)
  
  return(list(confusion_matrix = conf_matrix_class$table,
              overall_metrics = overall_metrics_class,
              class_metrics = class_metrics_class))
}

# Evaluate Random Forest and XGBoost
rf_evaluation_results <- evaluate_classification_model(classification_models$R.F, test_data_class, "Random Forest")
xgb_evaluation_results <- evaluate_classification_model(classification_models$XGB, test_data_class, "XGBoost")

# =============================================================================
# 8. PDF REPORT GENERATION (COMBINED WITH MULTIPLE REGRESSION MODELS)
# =============================================================================

# Stop PDF device
dev.off()
cat("All plots have been saved to combined_reg_classy_spotify_analysis_results_multiple_reg.pdf
     in your working directory.\n")