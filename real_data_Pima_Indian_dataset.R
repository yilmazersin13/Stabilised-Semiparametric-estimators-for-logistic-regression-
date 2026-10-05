#!/usr/bin/env Rscript
# Pima Indians Diabetes application (Section 6 of the paper)
# Outputs: tables_real/*.csv, figures_real/*.png

library(mlbench)   # only for the data set

# Functions

logit  <- function(p) log(p / (1 - p))
expit  <- function(x) 1 / (1 + exp(-x))
clip_p <- function(p, eps = 1e-8) pmin(pmax(p, eps), 1 - eps)

# Local linear smoother, Eq. (2.4) with q = 1 and Epanechnikov kernel
build_smoother <- function(t_vec, h) {
  n <- length(t_vec)
  S <- matrix(0, n, n)
  for (i in 1:n) {
    u <- (t_vec - t_vec[i]) / h
    w <- ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0)
    d <- t_vec - t_vec[i]
    sw   <- sum(w)
    swd  <- sum(w * d)
    swd2 <- sum(w * d^2)
    TwT <- matrix(c(sw, swd, swd, swd2), 2, 2)
    if (rcond(TwT) > 1e-14) {
      Tw <- rbind(w, w * d)
      S[i, ] <- solve(TwT, Tw)[1, ]
    }
  }
  S
}

# Nadaraya-Watson smoother with the same kernel, for the Kernel estimator of Roozbeh (2026)
build_nw_smoother <- function(t_vec, h) {
  n <- length(t_vec)
  S <- matrix(0, n, n)
  for (i in 1:n) {
    u <- (t_vec - t_vec[i]) / h
    w <- ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0)
    if (sum(w) > 0) S[i, ] <- w / sum(w)
  }
  S
}

# Main method: IWSRTE at fixed (h, k), Eqs. (2.10)-(2.12) and (3.1); k = 0 gives the IWSLSE
# beta_hat = solve(R_k) Xt' Om Vt,  R_k = Xt' Om Xt + k I,  Xt = (I - S) X,  Vt = (I - S) V
# f_hat = S (V - X beta_hat),  P = S + Xt solve(R_k) Xt' Om (I - S)
# sandwich variance: solve(R_k) Xt' A Xt solve(R_k),  A = Om (I - S) Om^{-1} (I - S)' Om
ridge_at_k <- function(y, X, S_h, k) {
  n <- nrow(X)
  p <- ncol(X)
  M  <- diag(n) - S_h
  Xt <- M %*% X

  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))

  for (s in 1:max_iter) {
    b_old <- beta
    om <- pmax(pi_hat * (1 - pi_hat), W_MIN)
    V  <- logit(pi_hat) + (y - pi_hat) / om
    Vt <- as.numeric(M %*% V)

    A  <- t(Xt) %*% (om * Xt)
    bv <- t(Xt) %*% (om * Vt)
    beta <- as.numeric(solve(A + k * diag(p), bv))

    f_hat  <- as.numeric(S_h %*% (V - X %*% beta))
    eta    <- as.numeric(X %*% beta + f_hat)
    pi_hat <- clip_p(expit(eta))

    if (sqrt(sum((beta - b_old)^2)) < tol) break
  }
  conv <- sqrt(sum((beta - b_old)^2)) < tol

  # Weights and working response at the final fit
  om <- pmax(pi_hat * (1 - pi_hat), W_MIN)
  V  <- logit(pi_hat) + (y - pi_hat) / om
  XtOX <- t(Xt) %*% (om * Xt)
  Rk <- XtOX + k * diag(p)
  P_hk <- S_h + Xt %*% (solve(Rk, t(Xt * om)) %*% M)

  # Xt' A Xt = C' Om^{-1} C with C = (I - S)' Om Xt
  C    <- crossprod(M, om * Xt)
  XtAX <- t(C) %*% (C / om)
  se   <- sqrt(pmax(diag(solve(Rk, XtAX) %*% solve(Rk)), 0))

  list(beta = beta, f_hat = f_hat, pi_hat = pi_hat, P_hk = P_hk,
       V = V, om = om, se = se, conv = conv,
       ok = all(is.finite(beta)) && conv)
}

# Deviance-based criteria, Eqs. (4.8)-(4.10)
sel_crit <- function(y, P_hk, pi_hat) {
  n    <- length(y)
  tr_P <- sum(diag(P_hk))
  pc   <- clip_p(pi_hat)
  D    <- -2 * sum(y * log(pc) + (1 - y) * log(1 - pc))
  c(GCV  = n * D / max(n - tr_P, 1)^2,
    AICc = D + 2 * tr_P + 2 * tr_P * (tr_P + 1) / max(n - tr_P - 1, 1),
    BIC  = D + log(n) * tr_P)
}

# Par-Ridge competitor; the intercept is not penalized
fit_par_ridge <- function(y, X, k = 0, intercept = FALSE) {
  if (intercept) X <- cbind(1, X)
  n <- nrow(X)
  p <- ncol(X)
  pen <- if (intercept) c(0, rep(k, p - 1)) else rep(k, p)

  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))

  for (s in 1:max_iter) {
    b_old <- beta
    om <- pmax(pi_hat * (1 - pi_hat), W_MIN)
    V  <- logit(pi_hat) + (y - pi_hat) / om
    beta <- as.numeric(solve(t(X) %*% (om * X) + diag(pen, p), t(X) %*% (om * V)))
    pi_hat <- clip_p(expit(as.numeric(X %*% beta)))
    if (sqrt(sum((beta - b_old)^2)) < tol) break
  }
  list(beta = beta, pi_hat = pi_hat)
}

# (h, k) search by MAIN_CRIT; only converged fits are eligible
select_hk <- function(y, X, t_vec, smoother_fun) {
  best <- list(gcv = Inf, h = NA, k = NA, ih = NA, fit = NULL)
  for (ih in seq_along(h_grid_real)) {
    S_h <- smoother_fun(t_vec, h_grid_real[ih])
    for (k in k_grid_real) {
      rk <- tryCatch(ridge_at_k(y, X, S_h, k), error = function(e) NULL)
      if (is.null(rk) || !rk$ok) next
      cr <- tryCatch(sel_crit(y, rk$P_hk, rk$pi_hat),
                     error = function(e) c(GCV = Inf, AICc = Inf, BIC = Inf))
      if (is.finite(cr[MAIN_CRIT]) && cr[MAIN_CRIT] < best$gcv) {
        best <- list(gcv = cr[MAIN_CRIT], h = h_grid_real[ih], k = k, ih = ih, fit = rk)
      }
    }
  }
  best
}

# Full-data metrics (in-sample, printed only)
compute_metrics <- function(y, pi_hat, beta, P_hk = NULL, label) {
  pi_c <- clip_p(pi_hat)
  dev <- -2 * sum(y * log(pi_c) + (1 - y) * log(1 - pi_c))
  mcr <- mean(as.integer(pi_hat > 0.5) != y)
  pos <- pi_hat[y == 1]
  neg <- pi_hat[y == 0]
  auc <- mean(outer(pos, neg, ">")) + 0.5 * mean(outer(pos, neg, "=="))
  edf <- if (!is.null(P_hk)) sum(diag(P_hk)) else length(beta)
  data.frame(Estimator = label, Deviance = round(dev, 2), EDF = round(edf, 2),
             MCR = round(mcr, 4), AUC = round(auc, 4), beta_norm = round(sqrt(sum(beta^2)), 3))
}

# Test-fold deviance, MCR and AUC (Table 4)
test_metrics <- function(y, pi_hat) {
  pi_c <- clip_p(pi_hat)
  pos <- pi_hat[y == 1]; neg <- pi_hat[y == 0]
  c(Deviance = -2 * sum(y * log(pi_c) + (1 - y) * log(1 - pi_c)),
    MCR = mean(as.integer(pi_hat > 0.5) != y),
    AUC = mean(outer(pos, neg, ">")) + 0.5 * mean(outer(pos, neg, "==")))
}

# f-hat at test ages by linear interpolation of the training fit (constant outside the training range)
pred_pi <- function(fit, X_te, t_tr, t_te) {
  f_te <- approx(t_tr, fit$f_hat, xout = t_te, rule = 2, ties = mean)$y
  expit(as.numeric(X_te %*% fit$beta) + f_te)
}

# Parameters

max_iter <- 100
tol <- 1e-6
K_FOLD    <- 5
MAIN_CRIT <- "BIC"
W_MIN     <- 1e-3
h_grid_real <- seq(0.08, 0.40, by = 0.03)    # 0.08, 0.11, ..., 0.38
k_grid_real <- c(seq(0.05, 5.0, by = 0.05), 7.5, 10, 15, 20, 30, 50, 100)
PR_INTERCEPT <- TRUE   # Par-Ridge has an intercept; the semiparametric fits have it in f
set.seed(2025)         # fold assignment

dir.create("figures_real", showWarnings = FALSE)
dir.create("tables_real",  showWarnings = FALSE)

# Data

data(PimaIndiansDiabetes)
dat <- PimaIndiansDiabetes

# Important: zeros in these biomarkers are impossible values; rows with any of them are removed (n = 768 -> 392)
bad_cols <- c("glucose", "pressure", "triceps", "insulin", "mass")
dat[bad_cols] <- lapply(dat[bad_cols], function(x) ifelse(x == 0, NA, x))
dat <- na.omit(dat)

y   <- as.integer(dat$diabetes == "pos")
age <- dat$age

X_base <- as.matrix(dat[, c(
  "pregnant", "glucose", "pressure", "triceps",
  "insulin", "mass", "pedigree"
)])

# Three engineered terms that create strong collinearity
X_aug <- cbind(
  X_base,
  glucose_sq = X_base[, "glucose"]^2,
  glu_x_ins  = X_base[, "glucose"] * X_base[, "insulin"],
  mass_x_tri = X_base[, "mass"] * X_base[, "triceps"]
)

X_raw <- X_aug
X <- scale(X_raw)
n <- nrow(X)
p <- ncol(X)
t_vec <- (age - min(age)) / (max(age) - min(age))

cat(sprintf("Sample size: n = %d\n", n))
cat(sprintf("Parametric dimension: p = %d\n", p))
cat(sprintf("Positive cases: %d (%.1f%%)\n", sum(y), 100 * mean(y)))

# Collinearity diagnostics (Figure 6 and Section 6.1 text)
R_cor <- cor(X_raw)
vif   <- diag(solve(R_cor))
ev    <- eigen(crossprod(X), symmetric = TRUE, only.values = TRUE)$values
kappa_X <- max(ev) / min(ev)
cat("\nVIF:\n"); print(round(vif, 2))
cat(sprintf("Covariates with VIF > 10: %d of %d\n", sum(vif > 10), p))
cat(sprintf("Condition number of X'X: %.1f\n", kappa_X))

# Full-data fits (Table 5 and Figure 7)

cat(sprintf("\nSelecting (h, k) via %s ...\n", MAIN_CRIT))
smoothers <- lapply(h_grid_real, function(h) build_smoother(t_vec, h))
best <- select_hk(y, X, t_vec, build_smoother)
if (is.na(best$ih) || is.null(best$fit)) {
  stop("No valid (h, k) was selected. Check the grids or convergence.")
}
cat(sprintf("Selected: h* = %.3f, k* = %.3f\n", best$h, best$k))

S_opt   <- smoothers[[best$ih]]
fit_rte <- best$fit
fit_lse <- ridge_at_k(y, X, S_opt, 0)
fit_pr  <- fit_par_ridge(y, X, best$k, PR_INTERCEPT)
best_ker <- select_hk(y, X, t_vec, build_nw_smoother)
fit_ker  <- best_ker$fit
cat(sprintf("Kernel: h* = %.3f, k* = %.3f\n", best_ker$h, best_ker$k))

metrics <- rbind(
  compute_metrics(y, fit_rte$pi_hat, fit_rte$beta, fit_rte$P_hk, "IWSRTE"),
  compute_metrics(y, fit_lse$pi_hat, fit_lse$beta, fit_lse$P_hk, "IWSLSE"),
  compute_metrics(y, fit_pr$pi_hat,  fit_pr$beta,  NULL,         "ParRidge"),
  compute_metrics(y, fit_ker$pi_hat, fit_ker$beta, fit_ker$P_hk, "KerIWSRE")
)
cat("\nIn-sample (full data) metrics, not cross-validated:\n")
print(metrics)
cat(sprintf("Converged: IWSRTE %s, IWSLSE %s, KerIWSRE %s\n", fit_rte$conv, fit_lse$conv, fit_ker$conv))

# Five-fold cross-validation (Table 4)
# Important: scaling, rescaling of age and (h, k) selection use the training fold only

folds  <- sample(rep(1:K_FOLD, length.out = n))
cv_res <- NULL
for (fo in 1:K_FOLD) {
  tr <- folds != fo; te <- folds == fo
  mu  <- colMeans(X_raw[tr, ]); sdv <- apply(X_raw[tr, ], 2, sd)
  X_tr <- scale(X_raw[tr, ], mu, sdv); X_te <- scale(X_raw[te, ], mu, sdv)
  a_min <- min(age[tr]); a_rng <- max(age[tr]) - min(age[tr])
  t_tr <- (age[tr] - a_min) / a_rng; t_te <- (age[te] - a_min) / a_rng

  b_rte <- select_hk(y[tr], X_tr, t_tr, build_smoother)
  b_ker <- select_hk(y[tr], X_tr, t_tr, build_nw_smoother)
  f_lse <- ridge_at_k(y[tr], X_tr, build_smoother(t_tr, b_rte$h), 0)
  f_pr  <- fit_par_ridge(y[tr], X_tr, b_rte$k, PR_INTERCEPT)
  X_te_pr <- if (PR_INTERCEPT) cbind(1, X_te) else X_te

  pis <- list(IWSRTE   = pred_pi(b_rte$fit, X_te, t_tr, t_te),
              IWSLSE   = pred_pi(f_lse,     X_te, t_tr, t_te),
              ParRidge = expit(as.numeric(X_te_pr %*% f_pr$beta)),
              KerIWSRE = pred_pi(b_ker$fit, X_te, t_tr, t_te))
  for (m in names(pis))
    cv_res <- rbind(cv_res, data.frame(fold = fo, Estimator = m, t(test_metrics(y[te], pis[[m]])),
                                       h = c(IWSRTE=b_rte$h, IWSLSE=b_rte$h, ParRidge=NA, KerIWSRE=b_ker$h)[m],
                                       k = c(IWSRTE=b_rte$k, IWSLSE=0, ParRidge=b_rte$k, KerIWSRE=b_ker$k)[m],
                                       conv_LSE = f_lse$conv))
  cat(sprintf("Fold %d done: h=%.2f k=%.2f (IWSRTE), h=%.2f k=%.2f (kernel), IWSLSE converged=%s\n",
              fo, b_rte$h, b_rte$k, b_ker$h, b_ker$k, f_lse$conv))
}
rownames(cv_res) <- NULL

# Summary tables

cv_tab <- aggregate(cbind(Deviance, MCR, AUC) ~ Estimator, data = cv_res, FUN = mean)
cv_tab$Deviance_total <- aggregate(Deviance ~ Estimator, data = cv_res, FUN = sum)$Deviance
cv_tab$AUC_sd <- aggregate(AUC ~ Estimator, data = cv_res, FUN = sd)$AUC
cat("\nFive-fold CV (means across folds; Deviance_total = sum over test folds):\n")
print(cv_tab, digits = 4)
write.csv(cv_res, "tables_real/T4_cv_folds.csv", row.names = FALSE)
write.csv(cv_tab, "tables_real/T4_cv.csv", row.names = FALSE)

coef_table <- data.frame(
  Covariate   = colnames(X),
  IWSRTE_beta = round(fit_rte$beta, 4),
  IWSRTE_SE   = round(fit_rte$se, 4),
  IWSRTE_OR   = round(exp(fit_rte$beta), 3),
  IWSLSE_beta = round(fit_lse$beta, 4),
  IWSLSE_SE   = round(fit_lse$se, 4),
  IWSLSE_OR   = round(exp(fit_lse$beta), 3)
)
cat("\nCoefficients (Table 5):\n")
print(coef_table)
write.csv(coef_table, "tables_real/T5_coef.csv", row.names = FALSE)

# Figures

# Figure 6: collinearity diagnostics and empirical age effect
png("figures_real/fig_diagnostics.png", width = 1400, height = 1100, res = 130)
par(mfrow = c(2, 2), mar = c(6.5, 6.5, 3, 1))

# (a) correlation matrix
nm <- colnames(X_raw)
image(1:p, 1:p, R_cor[, p:1], zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "",
      col = colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100),
      main = "(a) Correlation matrix")
axis(1, at = 1:p, labels = nm, las = 2, cex.axis = 0.7)
axis(2, at = 1:p, labels = rev(nm), las = 1, cex.axis = 0.7)
text(rep(1:p, p), rep(p:1, each = p), sprintf("%.2f", R_cor), cex = 0.5)

# (b) variance inflation factors
vif_col <- ifelse(vif > 10, "#B2182B", ifelse(vif > 5, "#F4A582", "gray70"))
bp <- barplot(vif, names.arg = nm, las = 2, col = vif_col, ylab = "VIF",
              ylim = c(0, max(vif) * 1.15), cex.names = 0.7, main = "(b) Variance inflation factors")
text(bp, vif, sprintf("%.1f", vif), pos = 3, cex = 0.6)
abline(h = c(5, 10), lty = 2, col = "gray40")
legend("topleft", legend = c("VIF > 10", "5 < VIF <= 10", "VIF <= 5"),
       fill = c("#B2182B", "#F4A582", "gray70"), bty = "n", cex = 0.75)

# (c) eigenvalue spectrum of X'X
plot(1:p, ev, type = "b", pch = 19, log = "y", col = "#2166AC",
     xlab = "Eigenvalue index", ylab = "Eigenvalue",
     main = sprintf("(c) Eigenvalue spectrum (kappa = %.0f)", kappa_X))

# (d) empirical diabetes proportion by 5-year age bin, with 95% CI and LOESS smoother
age_bin <- cut(age, breaks = seq(20, 85, by = 5), include.lowest = TRUE)
b_age <- tapply(age, age_bin, mean)
b_y   <- tapply(y, age_bin, mean)
b_n   <- tapply(y, age_bin, length)
b_se  <- sqrt(b_y * (1 - b_y) / b_n)
lo_fit <- loess(y ~ age)
a_seq  <- seq(min(age), max(age), length.out = 200)
plot(b_age, b_y, pch = 19, col = "#2166AC", ylim = c(0, 1),
     xlab = "Age (years)", ylab = "P(diabetes = positive)",
     main = "(d) Empirical age-diabetes relationship")
has_ci <- b_se > 0
arrows(b_age[has_ci], pmax(0, b_y - 1.96 * b_se)[has_ci], b_age[has_ci], pmin(1, b_y + 1.96 * b_se)[has_ci],
       angle = 90, code = 3, length = 0.03, col = "gray50")
lines(a_seq, predict(lo_fit, data.frame(age = a_seq)), lwd = 2, col = "#B2182B")
abline(h = mean(y), lty = 2, col = "gray40")
legend("topright", legend = c("Binned rate with 95% CI", "LOESS smoother",
                              sprintf("Overall rate = %.2f", mean(y))),
       col = c("#2166AC", "#B2182B", "gray40"), pch = c(19, NA, NA), lty = c(NA, 1, 2),
       bty = "n", cex = 0.75)
dev.off()
cat("Saved: figures_real/fig_diagnostics.png\n")

# Figure 7: f-hat on the logit scale with IWSLSE partial residuals V - X beta
pi_lse <- fit_lse$pi_hat
V_lse  <- logit(clip_p(pi_lse)) + (y - pi_lse) / (pi_lse * (1 - pi_lse))
partial_lse <- V_lse - as.numeric(X %*% fit_lse$beta)

ord <- order(age)

png("figures_real/fig_fhat_real.png", width = 800, height = 550, res = 120)
par(mar = c(4.5, 4.5, 3, 1))
ylims <- range(c(partial_lse, fit_rte$f_hat, fit_lse$f_hat), na.rm = TRUE, finite = TRUE)
ylims <- ylims + c(-0.1, 0.1) * diff(ylims)
plot(age, partial_lse,
     pch = 16, cex = 0.7,
     col = adjustcolor("gray40", 0.35),
     xlab = "Age (years)",
     ylab = expression("Partial residuals  V" - X * hat(beta)),
     main = "Estimated nonparametric component",
     ylim = ylims)
lines(age[ord], fit_lse$f_hat[ord], col = "#B2182B", lwd = 2.5, lty = 2)
lines(age[ord], fit_rte$f_hat[ord], col = "#2166AC", lwd = 2.8, lty = 1)
abline(h = 0, col = "gray70", lty = 3)
legend("topright",
       legend = c(bquote("IWSRTE  (k = " * .(round(best$k, 2)) * ")"),
                  "IWSLSE  (k = 0)",
                  "Partial residuals"),
       col = c("#2166AC", "#B2182B", adjustcolor("gray40", 0.6)),
       lty = c(1, 2, NA),
       pch = c(NA, NA, 16),
       lwd = c(2.8, 2.5, NA),
       bty = "n", cex = 0.85)
dev.off()
cat("Saved: figures_real/fig_fhat_real.png\n")
