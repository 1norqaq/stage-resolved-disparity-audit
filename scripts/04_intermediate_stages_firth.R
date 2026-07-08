#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(logistf)
})

option_list <- list(
  make_option(c("--input"), type="character"),
  make_option(c("--out"), type="character")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$input) || is.null(opt$out)) {
  stop("Usage: Rscript scripts/04_intermediate_stages_firth.R --input data/linked_applications.csv --out results/tables/intermediate_stages_firth.csv")
}

df <- read_csv(opt$input, show_col_types = FALSE)
disadvantaged_groups <- c("Muslim", "African", "Indian-subcontinent")
milestones <- c("prequalified", "interview_shortlist", "offered")
milestones <- milestones[milestones %in% names(df)]

rows <- list()
for (m in milestones) {
  work <- df %>%
    mutate(disadvantaged = as.integer(ethnicity %in% disadvantaged_groups)) %>%
    filter(!is.na(.data[[m]]), !is.na(fit_q), !is.na(gender))
  f <- as.formula(paste0(m, " ~ disadvantaged + fit_q + factor(gender)"))
  fit <- logistf(f, data = work)
  beta <- coef(fit)["disadvantaged"]
  ci <- confint(fit)["disadvantaged",]
  rows[[m]] <- data.frame(
    milestone = m,
    estimate = beta,
    or = exp(beta),
    ci_low = exp(ci[1]),
    ci_high = exp(ci[2]),
    n = nrow(work)
  )
}

out <- bind_rows(rows)
dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)
write_csv(out, opt$out)
