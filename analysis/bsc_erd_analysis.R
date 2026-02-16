# ============================================================================
# ERD ANALYSIS AND VISUALISATION
# ============================================================================

#install and load packages
# install.packages(c("gtsummary", "gt", "flextable"))
library(lme4,
       lmerTest,
       emmeans,
       effectsize,
       tidyverse,
       gtsummary,
       gt,
       flextable,
       patchwork )

#lmm analysis function
analyze_erd_lmm <- function(data_file, event_type, band_name) {

  df <- read.csv(data_file)
  
  #convert to factors + reference levels
  df$participant_id <- factor(df$participant_id)
  df$social <- factor(df$social, levels = c("Non-social", "Social"))
  df$uncertainty <- factor(df$uncertainty, 
                          levels = c("Certain", "Low Uncertain", "High Uncertain"))
  df$bin_idx <- as.numeric(df$bin_idx)
  
  #check if task column exists and has variation
  has_task <- "task" %in% names(df) && length(unique(df$task)) > 1
  
  #model formula
  if (has_task) {
    df$task <- factor(df$task, levels = c("No-go", "Go"))
    formula <- erd_value ~ social * uncertainty * task * bin_idx + (1 | participant_id)
  } else {
    formula <- erd_value ~ social * uncertainty * bin_idx + (1 | participant_id)
  }

  model <- lmer(formula, data = df, REML = TRUE)
  
  #anova iii
  anova_result <- anova(model)
  print(anova_result)
  
  #partial eta2
  anova_df <- as.data.frame(anova_result)
  eta_squared_df <- effectsize::eta_squared(model, partial = TRUE, ci = 0.95)
  
  #merge effect sizes with ANOVA results
  #match effect names
  anova_df$Effect <- rownames(anova_df)
  eta_squared_df$Effect <- eta_squared_df$Parameter
  
  #join dataframes
  anova_df <- anova_df %>%
    left_join(eta_squared_df %>% select(Effect, Eta2_partial, CI_low, CI_high), 
              by = "Effect")
  
  anova_df <- anova_df %>% #renaming variables
    rename(partial_eta_sq = Eta2_partial,
           eta_sq_CI_low = CI_low,
           eta_sq_CI_high = CI_high)
  
  #print results
  for (i in 1:nrow(anova_df)) {
    effect_name <- anova_df[i, "Effect"]
    f_val <- anova_df[i, "F value"]
    p_val <- anova_df[i, "Pr(>F)"]
    df_num <- anova_df[i, "NumDF"]
    df_denom <- anova_df[i, "DenDF"]
    eta_sq <- anova_df[i, "partial_eta_sq"]
    eta_ci_low <- anova_df[i, "eta_sq_CI_low"]
    eta_ci_high <- anova_df[i, "eta_sq_CI_high"]    
  }
  
  results <- list(
    model = model,
    anova = anova_result,
    effect_sizes = anova_df,
    data = df
  )
  
  return(results)
}

#function to run all analyses
run_all_erd_analyses <- function(data_dir = "../erd_data_for_r", 
                                 output_file = "erd_lmm_results.csv") {
  
  bands <- c("alpha", "low_beta", "high_beta")
  epochs <- c("cue", "move")
  
  all_results <- list()
  summary_df <- data.frame()
  
  for (band in bands) {
    for (epoch in epochs) {
      
      #construct filename
      if (epoch == "cue") {
        filename <- file.path(data_dir, sprintf("erd_%s_%s_C3.csv", epoch, band))
      } else {
        filename <- file.path(data_dir, sprintf("erd_%s_%s_C3_Go.csv", epoch, band))
      }

      #analyse
      tryCatch({
        results <- analyze_erd_lmm(filename, epoch, band)

        key <- paste(epoch, band, sep = "_")
        all_results[[key]] <- results
        
        #effect sizes
        effect_sizes <- results$effect_sizes
        effect_sizes$Band <- band
        effect_sizes$Epoch <- epoch
        effect_sizes$Effect <- rownames(effect_sizes)
        
        summary_df <- rbind(summary_df, effect_sizes)
        
      }, error = function(e) {
        cat("ERROR analyzing", epoch, band, ":", conditionMessage(e), "\n\n")
      })
    }
  }
  
  #reorder columns
  summary_df <- summary_df %>%
    select(Band, Epoch, Effect, `F value`, NumDF, DenDF, 
           `Pr(>F)`, partial_eta_sq, everything())
  
  #save summary to csv
  write.csv(summary_df, output_file, row.names = FALSE)
  return(list(
    results = all_results,
    summary = summary_df
  ))
}

#run al analyses
all_results <- run_all_erd_analyses(
  data_dir = "../erd_data_for_r",
  output_file = "erd_lmm_results.csv"
)

#coefficients and 95ci
extract_coefficients <- function(model, conf_level = 0.95) {
  
  cat("\n", rep("=", 80), "\n", sep="")
  cat("FIXED EFFECTS COEFFICIENTS\n")
  cat(rep("=", 80), "\n\n", sep="")
  
  #coef summary
  coef_summary <- summary(model)$coefficients
  
  #calculate ci
  ci <- confint(model, parm = "beta_", method = "Wald", level = conf_level)
  
  #data frame
  coef_df <- data.frame(
    Predictor = rownames(coef_summary),
    Beta = coef_summary[, "Estimate"],
    SE = coef_summary[, "Std. Error"],
    df = coef_summary[, "df"],
    t_value = coef_summary[, "t value"],
    p_value = coef_summary[, "Pr(>|t|)"],
    CI_lower = ci[, 1],
    CI_upper = ci[, 2],
    stringsAsFactors = FALSE
  )
  
  #significance starts for p values
  coef_df$sig <- ifelse(coef_df$p_value < 0.001, "***",
                 ifelse(coef_df$p_value < 0.01, "**",
                 ifelse(coef_df$p_value < 0.05, "*", "")))
  return(coef_df)
}

#example of extracting coefficients for one model
#move alpha model
move_alpha_model <- all_results$results$move_alpha$model
move_alpha_coefs <- extract_coefficients(move_alpha_model)
print(move_alpha_coefs)

#post-hoc emmeans for significant effects for move alpha model
emm_uncertainty_move_alpha <- emmeans(move_alpha_model, ~ uncertainty)
pairs_uncertainty_move_alpha <- pairs(emm_uncertainty_move_alpha, adjust = "tukey")
print(pairs_uncertainty_move_alpha)

pairs_uncertainty_df <- as.data.frame(pairs_uncertainty_move_alpha)
effect_sizes_uncertainty_move_alpha <- eff_size(emm_uncertainty_move_alpha, sigma = sigma(move_alpha_model), edf = df.residual(move_alpha_model))
print(effect_sizes_uncertainty_move_alpha)

#function to prepare data for gtsummary
prepare_anova_for_gtsummary <- function(results_list, epoch = "cue") {
  
  bands <- c("alpha", "low_beta", "high_beta")
  all_effects <- list()
  
  for (band in bands) {
    key <- paste(epoch, band, sep = "_")
    
    if (key %in% names(results_list$results)) {
      df <- results_list$results[[key]]$effect_sizes
      
      #create formatted data
      band_data <- data.frame(
        Effect = df$Effect,
        Band = case_when(
          band == "alpha" ~ "Alpha",
          band == "low_beta" ~ "Low Beta",
          band == "high_beta" ~ "High Beta"
        ),
        F_value = df$`F value`,
        df_num = df$NumDF,
        df_denom = df$DenDF,
        p_value = df$`Pr(>F)`,
        partial_eta_sq = df$partial_eta_sq,
        stringsAsFactors = FALSE
      )
      
      all_effects[[band]] <- band_data
    }
  }
  combined <- bind_rows(all_effects) #combine all bands
  combined <- combined %>%
    mutate(
      Effect_Category = case_when(
        !grepl(":", Effect) ~ "Main Effects",
        str_count(Effect, ":") == 1 ~ "Two-Way Interactions",
        str_count(Effect, ":") == 2 ~ "Three-Way Interactions",
        str_count(Effect, ":") == 3 ~ "Four-Way Interaction"
      ),
      #format effect names
      Effect_Name = str_replace_all(Effect, ":", " × ")
    )
  
  return(combined)
}

#function for creating the results table
create_custom_anova_table <- function(results_list, epoch = "cue") {
  
  # Prepare data
  df <- prepare_anova_for_gtsummary(results_list, epoch)
  
  df_wide <- df %>% #changing the table formatting
    mutate(
      #format values
      F_formatted = sprintf("%.2f", F_value),
      df_formatted = sprintf("%d, %.0f", df_num, df_denom),
      p_formatted = case_when(
        p_value < 0.001 ~ "< 0.001***",
        p_value < 0.008 ~ sprintf("%.3f**", p_value),
        p_value < 0.05 ~ sprintf("%.3f*", p_value),
        TRUE ~ sprintf("%.3f", p_value)
      ),
      eta_formatted = ifelse(partial_eta_sq < 0.001, 
                            "< 0.001", 
                            sprintf("%.3f", partial_eta_sq))
    ) %>%
    select(Effect_Category, Effect_Name, Band, 
           F_formatted, df_formatted, p_formatted, eta_formatted) %>%
    pivot_wider(
      names_from = Band,
      values_from = c(F_formatted, df_formatted, p_formatted, eta_formatted),
      names_glue = "{Band}_{.value}"
    )
  
  #gt table
  tbl <- df_wide %>%
    gt(groupname_col = "Effect_Category") %>%
    
    #title and subtitle
    tab_header(
      title = paste("Type III ANOVA Results:", 
                   ifelse(epoch == "cue", "Cue Epoch", "Movement Epoch")),
      subtitle = "Fixed Effects Tests with Partial η²"
    ) %>%
    
    #rename columns with proper formatting
    cols_label(
      Effect_Name = "Effect",
      Alpha_F_formatted = "F",
      Alpha_df_formatted = "df",
      Alpha_p_formatted = "p",
      Alpha_eta_formatted = "η²ₚ",
      `Low Beta_F_formatted` = "F",
      `Low Beta_df_formatted` = "df",
      `Low Beta_p_formatted` = "p",
      `Low Beta_eta_formatted` = "η²ₚ",
      `High Beta_F_formatted` = "F",
      `High Beta_df_formatted` = "df",
      `High Beta_p_formatted` = "p",
      `High Beta_eta_formatted` = "η²ₚ"
    ) %>%
    
    #spanning column headers
    tab_spanner(
      label = "Alpha Band (7-13 Hz)",
      columns = starts_with("Alpha")
    ) %>%
    tab_spanner(
      label = "Low Beta Band (13-25 Hz)",
      columns = starts_with("Low Beta")
    ) %>%
    tab_spanner(
      label = "High Beta Band (25-30 Hz)",
      columns = starts_with("High Beta")
    ) %>%
    
    #style the table
    tab_style(
      style = cell_text(weight = "bold", style = "italic"),
      locations = cells_column_spanners()
    ) %>%
    tab_style(
      style = cell_text(weight = "bold", align = "center"),
      locations = cells_column_labels()
    ) %>%
    tab_style(
      style = cell_text(style = "italic"),
      locations = cells_row_groups()
    ) %>%
    tab_options(
      table.font.size = px(10),
      heading.title.font.size = px(14),
      heading.subtitle.font.size = px(12),
      row_group.font.weight = "bold",
      column_labels.font.weight = "bold",
      column_labels.padding = px(5)
    )
  
  return(tbl)
}

#tables for cue and move epochs
table_cue <- create_custom_anova_table(all_results, epoch = "cue")
table_move <- create_custom_anova_table(all_results, epoch = "move")

table_cue
table_move

#save as HTML
gtsave(table_cue, "Table1_Cue_ANOVA.html")
gtsave(table_move, "Table2_Move_ANOVA.html")

# ============================================================================
# VISUALISATION
# ============================================================================

#function for an erd plot
create_erd_plot <- function(results_list, epoch = "cue", band = "alpha") {
  
  key <- paste(epoch, band, sep = "_")
  
  if (!key %in% names(results_list$results)) {
    stop("Results not found for ", epoch, " ", band)
  }

  df <- results_list$results[[key]]$data
  
  #calculate mean and se for each condition
  plot_data <- df %>%
    group_by(social, uncertainty, bin_idx) %>%
    summarise(
      mean_erd = mean(erd_value, na.rm = TRUE),
      se_erd = sd(erd_value, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  #new title
  band_name <- case_when(
    band == "alpha" ~ "Alpha (7-13 Hz)",
    band == "low_beta" ~ "Low Beta (13-25 Hz)",
    band == "high_beta" ~ "High Beta (25-30 Hz)"
  )
  
  epoch_name <- ifelse(epoch == "cue", "Cue Phase", "Movement Phase")
  
  #plot
  p <- ggplot(plot_data, aes(x = bin_idx, y = mean_erd, 
                              color = interaction(social, uncertainty),
                              linetype = social)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean_erd - se_erd, 
                    ymax = mean_erd + se_erd,
                    fill = interaction(social, uncertainty)),
                alpha = 0.2, color = NA) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(
      name = "Condition",
      values = c(
        "Non-social.Certain" = "#1f77b4",
        "Non-social.Low Uncertain" = "#ff7f0e",
        "Non-social.High Uncertain" = "#2ca02c",
        "Social.Certain" = "#d62728",
        "Social.Low Uncertain" = "#9467bd",
        "Social.High Uncertain" = "#8c564b"
      ),
      labels = c(
        "Non-social.Certain" = "Non-social, Certain",
        "Non-social.Low Uncertain" = "Non-social, Low Uncertain",
        "Non-social.High Uncertain" = "Non-social, High Uncertain",
        "Social.Certain" = "Social, Certain",
        "Social.Low Uncertain" = "Social, Low Uncertain",
        "Social.High Uncertain" = "Social, High Uncertain"
      )
    ) +
    scale_fill_manual(
      name = "Condition",
      values = c(
        "Non-social.Certain" = "#1f77b4",
        "Non-social.Low Uncertain" = "#ff7f0e",
        "Non-social.High Uncertain" = "#2ca02c",
        "Social.Certain" = "#d62728",
        "Social.Low Uncertain" = "#9467bd",
        "Social.High Uncertain" = "#8c564b"
      ),
      labels = c(
        "Non-social.Certain" = "Non-social, Certain",
        "Non-social.Low Uncertain" = "Non-social, Low Uncertain",
        "Non-social.High Uncertain" = "Non-social, High Uncertain",
        "Social.Certain" = "Social, Certain",
        "Social.Low Uncertain" = "Social, Low Uncertain",
        "Social.High Uncertain" = "Social, High Uncertain"
      )
    ) +
    scale_linetype_manual(
      name = "Social Context",
      values = c("Non-social" = 1, "Social" = 2)
    ) +
    scale_x_continuous(
      breaks = unique(plot_data$bin_idx),
      labels = unique(plot_data$bin_idx)
    ) +
    labs(
      title = paste(band_name, "-", epoch_name),
      x = "Time Bin",
      y = "ERDS Value (% change)",
    #   caption = "Shaded areas represent ±1 SE"
    ) +
    theme_minimal(base_size = 16) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 14),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 16)
    )
  
  return(p)
}

#function to create a combined panel of all plots 
create_erd_panel <- function(results_list) {
  
  bands <- c("alpha", "low_beta", "high_beta")
  epochs <- c("cue", "move")
  
  plots <- list()
  
  for (epoch in epochs) {
    for (band in bands) {
      key <- paste(epoch, band, sep = "_")
      if (key %in% names(results_list$results)) {
        p <- create_erd_plot(results_list, epoch, band)
        plots[[key]] <- p
      }
    }
  }
  
  #remove axis labels from inside plots
  plots$cue_alpha <- plots$cue_alpha + labs(x = NULL)
  plots$cue_low_beta <- plots$cue_low_beta + labs(x = NULL, y = NULL)
  plots$cue_high_beta <- plots$cue_high_beta + labs(x = NULL, y = NULL)
  plots$move_low_beta <- plots$move_low_beta + labs(y = NULL)
  plots$move_high_beta <- plots$move_high_beta + labs(y = NULL)
  
  combined_plot <- (plots$cue_alpha | plots$cue_low_beta | plots$cue_high_beta) /
                   (plots$move_alpha | plots$move_low_beta | plots$move_high_beta) +
    plot_layout(guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 14)
    )
  
  combined_plot <- combined_plot +
    plot_annotation(
    #   title = "Event-Related Desynchronization (ERD) Across Conditions",
    #   subtitle = "Top row: Cue Epoch | Bottom row: Movement Epoch",
      theme = theme(
        plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
        plot.subtitle = element_text(size = 16, hjust = 0.5)
      )
    )
  
  return(combined_plot)
}

#plots for each band and epoch
plot_cue_alpha <- create_erd_plot(all_results, epoch = "cue", band = "alpha")
plot_cue_low_beta <- create_erd_plot(all_results, epoch = "cue", band = "low_beta")
plot_cue_high_beta <- create_erd_plot(all_results, epoch = "cue", band = "high_beta")

plot_move_alpha <- create_erd_plot(all_results, epoch = "move", band = "alpha")
plot_move_low_beta <- create_erd_plot(all_results, epoch = "move", band = "low_beta")
plot_move_high_beta <- create_erd_plot(all_results, epoch = "move", band = "high_beta")

#combined panel
combined_panel <- create_erd_panel(all_results)

#display plots
print(plot_cue_alpha)
print(plot_cue_low_beta)
print(plot_cue_high_beta)
print(plot_move_alpha)
print(plot_move_low_beta)
print(plot_move_high_beta)
print(combined_panel)

#save plots
ggsave("ERD_Cue_Alpha.png", plot_cue_alpha, width = 10, height = 6, dpi = 300)
ggsave("ERD_Cue_LowBeta.png", plot_cue_low_beta, width = 10, height = 6, dpi = 300)
ggsave("ERD_Cue_HighBeta.png", plot_cue_high_beta, width = 10, height = 6, dpi = 300)
ggsave("ERD_Move_Alpha.png", plot_move_alpha, width = 10, height = 6, dpi = 300)
ggsave("ERD_Move_LowBeta.png", plot_move_low_beta, width = 10, height = 6, dpi = 300)
ggsave("ERD_Move_HighBeta.png", plot_move_high_beta, width = 10, height = 6, dpi = 300)

ggsave("ERD_Combined_Panel.png", combined_panel, width = 18, height = 10, dpi = 800)

