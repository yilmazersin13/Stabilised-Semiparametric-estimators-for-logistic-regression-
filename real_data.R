# ==============================================================
# Section 7.1 — Multicollinearity diagnostics for Pima (augmented)
# ==============================================================

library(mlbench)
data(PimaIndiansDiabetes)
dat <- PimaIndiansDiabetes

# Remove physiologically implausible zeros
bad_cols <- c("glucose", "pressure", "triceps", "insulin", "mass")
dat[bad_cols] <- lapply(dat[bad_cols], function(x) ifelse(x == 0, NA, x))
dat <- na.omit(dat)  # n = 392

y <- as.integer(dat$diabetes == "pos")
age <- dat$age

# ── Augmented design matrix (p = 10) ─────────────────────────
X_base <- as.matrix(dat[, c("pregnant","glucose","pressure","triceps",
                            "insulin","mass","pedigree")])

X_aug <- cbind(
  X_base,
  glucose_sq = X_base[,"glucose"]^2,
  glu_x_ins  = X_base[,"glucose"] * X_base[,"insulin"],
  mass_x_tri = X_base[,"mass"] * X_base[,"triceps"]
)

X_raw <- X_aug
X <- scale(X_raw)

cat(sprintf("Sample size: n = %d\n", nrow(X)))
cat(sprintf("Parametric dimension: p = %d\n", ncol(X)))
cat(sprintf("Positive cases: %d (%.1f%%)\n", sum(y), 100*mean(y)))

dir.create("figures_real", showWarnings = FALSE)
dir.create("tables_real", showWarnings = FALSE)

# ── Multicollinearity diagnostics ────────────────────────────

corr_mat <- cor(X)
XtX <- t(X) %*% X
eig_vals <- eigen(XtX)$values
cond_num <- max(eig_vals) / min(eig_vals)
cond_idx <- sqrt(cond_num)

p <- ncol(X)
vifs <- numeric(p)
names(vifs) <- colnames(X)
for (j in 1:p) {
  fit <- lm(X[,j] ~ X[,-j])
  vifs[j] <- 1 / (1 - summary(fit)$r.squared)
}

cat("\n--- Diagnostics ---\n")
cat(sprintf("Condition number of X'X: %.2f\n", cond_num))
cat(sprintf("Condition index:         %.2f\n", cond_idx))
cat(sprintf("Largest eigenvalue:      %.2f\n", max(eig_vals)))
cat(sprintf("Smallest eigenvalue:     %.4f\n", min(eig_vals)))
cat("\nVariance Inflation Factors:\n")
print(round(vifs, 2))

# Save table
diag_table <- data.frame(
  Covariate = colnames(X),
  Mean = round(colMeans(X_raw), 2),
  SD   = round(apply(X_raw, 2, sd), 2),
  VIF  = round(vifs, 2)
)
write.csv(diag_table, "tables_real/table_real_diagnostics.csv", row.names = FALSE)

# ── 2x2 diagnostic figure ────────────────────────────────────

png("figures_real/fig_diagnostics.png", width = 1100, height = 950, res = 120)
par(mfrow = c(2, 2), mar = c(5, 4.5, 3, 1.5), oma = c(0, 0, 1, 0))

# (a) Correlation heatmap
image(1:p, 1:p, corr_mat[, p:1],
      col = colorRampPalette(c("#2166AC","#F7F7F7","#B2182B"))(100),
      zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "",
      main = "(a) Correlation matrix")
axis(1, at = 1:p, labels = colnames(X), las = 2, cex.axis = 0.65)
axis(2, at = 1:p, labels = rev(colnames(X)), las = 1, cex.axis = 0.65)
for (i in 1:p) for (j in 1:p) {
  text(i, p-j+1, sprintf("%.2f", corr_mat[i,j]), cex = 0.55,
       col = ifelse(abs(corr_mat[i,j]) > 0.5, "white", "black"))
}

# (b) VIFs
vif_colors <- ifelse(vifs > 10, "#B2182B",
                     ifelse(vifs > 5, "#F4A582", "#92C5DE"))
bp <- barplot(vifs, col = vif_colors, las = 2, cex.names = 0.65,
              ylab = "VIF", main = "(b) Variance inflation factors",
              ylim = c(0, max(vifs)*1.25))
abline(h = c(5, 10), col = "gray40", lty = c(3, 2), lwd = 1.2)
text(bp, vifs, sprintf("%.1f", vifs), pos = 3, cex = 0.65)
legend("topright", legend = c("VIF > 10", "5 < VIF ≤ 10", "VIF ≤ 5"),
       fill = c("#B2182B", "#F4A582", "#92C5DE"), bty = "n", cex = 0.7)

# (c) Eigenvalue spectrum
plot(1:p, eig_vals, type = "b", pch = 19, col = "#2166AC", lwd = 2,
     xlab = "Eigenvalue index", ylab = "Eigenvalue", log = "y",
     main = sprintf("(c) Eigenvalue spectrum (κ = %.0f)", cond_num))
abline(h = 1, col = "gray50", lty = 3)

# (d) Nonlinear age effect
age_bins   <- cut(age, breaks = seq(20, 85, by = 5), include.lowest = TRUE)
age_midpts <- sapply(split(age, age_bins), mean)
prop_diab  <- sapply(split(y, age_bins), mean)
n_per_bin  <- sapply(split(y, age_bins), length)
se_bin <- sqrt(prop_diab * (1 - prop_diab) / pmax(n_per_bin, 1))
lo <- pmax(0, prop_diab - 1.96*se_bin)
hi <- pmin(1, prop_diab + 1.96*se_bin)

loess_fit  <- loess(y ~ age, span = 0.7)
age_seq    <- seq(min(age), max(age), length.out = 200)
loess_pred <- predict(loess_fit, newdata = data.frame(age = age_seq))

plot(age_midpts, prop_diab, type = "n",
     xlab = "Age (years)", ylab = "P(diabetes = positive)",
     main = "(d) Empirical age–diabetes relationship",
     ylim = c(0, 0.8))
arrows(age_midpts, lo, age_midpts, hi, angle = 90, code = 3,
       length = 0.03, col = "gray60")
points(age_midpts, prop_diab, pch = 19, col = "#2166AC", cex = 1.2)
lines(age_seq, loess_pred, col = "#B2182B", lwd = 2.5)
abline(h = mean(y), lty = 3, col = "gray40")
legend("topright",
       legend = c("5-year bin ± 95% CI", "LOESS smoother",
                  sprintf("Overall rate = %.2f", mean(y))),
       col = c("#2166AC", "#B2182B", "gray40"),
       lty = c(NA, 1, 3), pch = c(19, NA, NA),
       lwd = c(NA, 2.5, 1), bty = "n", cex = 0.7)

dev.off()
cat("\nFigure saved to figures_real/fig_diagnostics.png\n")




# ==============================================================
# Section 7.2 — Real Data Results (Pima, augmented)
# Assumes: simulation functions are already loaded
#          (build_smoother, irls_converge, ridge_at_k, sel_crit,
#           fit_par_ridge, logit, expit, clip_p)
# Assumes: y, X, age loaded from Section 7.1
# ==============================================================

# Scale age to [0,1] for the nonparametric component
t_vec <- (age - min(age)) / (max(age) - min(age))
n <- nrow(X); p <- ncol(X)

# Grids for (h, k) selection
h_grid_real <- seq(0.08, 0.40, by = 0.03)
k_grid_real <- seq(0, 2.0, by = 0.05)

dir.create("figures_real", showWarnings = FALSE)
dir.create("tables_real",      showWarnings = FALSE)

# ── 1. GCV-optimal (h, k) for IWSRTE ─────────────────────────

cat("Selecting (h, k) via GCV ...\n")

best <- list(gcv = Inf, h = NA, k = NA, ih = NA, fit = NULL)
smoothers <- lapply(h_grid_real, function(h) build_smoother(t_vec, h))

for (ih in seq_along(h_grid_real)) {
  cf <- tryCatch(irls_converge(y, X, smoothers[[ih]]), error = function(e) NULL)
  if (is.null(cf)) next
  for (k in k_grid_real) {
    rk <- tryCatch(ridge_at_k(y, X, smoothers[[ih]], k), error = function(e) NULL)
    if (is.null(rk) || !rk$ok) next
    cr <- tryCatch(sel_crit(y, rk$P_hk), error = function(e) c(GCV = Inf))
    if (is.finite(cr["GCV"]) && cr["GCV"] < best$gcv) {
      best <- list(gcv = cr["GCV"], h = h_grid_real[ih], k = k,
                   ih = ih, fit = rk)
    }
  }
}

cat(sprintf("Selected: h* = %.3f, k* = %.3f\n", best$h, best$k))
S_opt   <- smoothers[[best$ih]]
fit_rte <- best$fit
fit_lse <- ridge_at_k(y, X, S_opt, 0)       # IWSLSE
fit_pr  <- fit_par_ridge(y, X, best$k)       # parametric ridge at same k

# ── 2. Performance metrics for each estimator ────────────────

compute_metrics <- function(y, pi_hat, beta, P_hk = NULL, label) {
  pi_c <- clip_p(pi_hat)
  dev  <- -2 * sum(y * log(pi_c) + (1 - y) * log(1 - pi_c))
  mcr  <- mean(as.integer(pi_hat > 0.5) != y)
  # AUC via Mann–Whitney
  pos <- pi_hat[y == 1]; neg <- pi_hat[y == 0]
  auc <- mean(outer(pos, neg, ">")) + 0.5 * mean(outer(pos, neg, "=="))
  # Effective df
  edf <- if (!is.null(P_hk)) sum(diag(P_hk)) else length(beta)
  data.frame(Estimator = label,
             Deviance  = round(dev, 2),
             EDF       = round(edf, 2),
             MCR       = round(mcr, 4),
             AUC       = round(auc, 4),
             beta_norm = round(sqrt(sum(beta^2)), 3))
}

metrics <- rbind(
  compute_metrics(y, fit_rte$pi_hat, fit_rte$beta, fit_rte$P_hk, "IWSRTE"),
  compute_metrics(y, fit_lse$pi_hat, fit_lse$beta, fit_lse$P_hk, "IWSLSE"),
  compute_metrics(y, fit_pr$pi_hat,  fit_pr$beta,  NULL,          "ParRidge")
)
print(metrics)
write.csv(metrics, "tables_real/table_real_performance.csv", row.names = FALSE)

# ── 3. Coefficient table with SEs from Eq. (4.1) ─────────────

# Asymptotic covariance matrix: Σ_k = R_k^{-1} X'ΩX R_k^{-1}
om_rte <- fit_rte$pi_hat * (1 - fit_rte$pi_hat)
M  <- diag(n) - S_opt
Xt <- M %*% X
XtOX_rte <- t(Xt) %*% (om_rte * Xt)
Rk_rte <- XtOX_rte + best$k * diag(p)
Sigma_rte <- solve(Rk_rte) %*% XtOX_rte %*% solve(Rk_rte)
se_rte <- sqrt(diag(Sigma_rte))

om_lse <- fit_lse$pi_hat * (1 - fit_lse$pi_hat)
XtOX_lse <- t(Xt) %*% (om_lse * Xt)
Sigma_lse <- solve(XtOX_lse)
se_lse <- sqrt(diag(Sigma_lse))

coef_table <- data.frame(
  Covariate   = colnames(X),
  IWSRTE_beta = round(fit_rte$beta, 4),
  IWSRTE_SE   = round(se_rte, 4),
  IWSRTE_OR   = round(exp(fit_rte$beta), 3),
  IWSLSE_beta = round(fit_lse$beta, 4),
  IWSLSE_SE   = round(se_lse, 4),
  IWSLSE_OR   = round(exp(fit_lse$beta), 3)
)
write.csv(coef_table, "tables_real/table_real_coefficients.csv", row.names = FALSE)
print(coef_table)

# ── 4. Figure: fitted f(age) + partial residuals ─────────────

# Partial residuals: V_i - x_i'β  (working response minus parametric fit)
pi_rte <- fit_rte$pi_hat
V_rte  <- logit(clip_p(pi_rte)) + (y - pi_rte) / (pi_rte * (1 - pi_rte))
partial_rte <- V_rte - as.numeric(X %*% fit_rte$beta)

pi_lse <- fit_lse$pi_hat
V_lse  <- logit(clip_p(pi_lse)) + (y - pi_lse) / (pi_lse * (1 - pi_lse))
partial_lse <- V_lse - as.numeric(X %*% fit_lse$beta)

ord <- order(age)

png("figures_real/fig_fhat_real.png", width = 1000, height = 480, res = 120)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))

# IWSRTE panel
plot(age, partial_rte, pch = 16, col = adjustcolor("black", 0.35), cex = 0.6,
     xlab = "Age (years)", ylab = "Partial residuals  V − Xβ̂",
     main = sprintf("IWSRTE (k = %.2f)", best$k),
     ylim = range(c(partial_rte, partial_lse, fit_rte$f_hat, fit_lse$f_hat)))
lines(age[ord], fit_rte$f_hat[ord], col = "#2166AC", lwd = 2.5)
abline(h = 0, col = "gray60", lty = 3)
legend("topright", legend = c(expression(hat(f)(age)), "Partial residuals"),
       col = c("#2166AC", "black"), lty = c(1, NA), pch = c(NA, 16),
       lwd = c(2.5, NA), bty = "n", cex = 0.8)

# IWSLSE panel
plot(age, partial_lse, pch = 16, col = adjustcolor("black", 0.35), cex = 0.6,
     xlab = "Age (years)", ylab = "Partial residuals  V − Xβ̂",
     main = "IWSLSE (k = 0)",
     ylim = range(c(partial_rte, partial_lse, fit_rte$f_hat, fit_lse$f_hat)))
lines(age[ord], fit_lse$f_hat[ord], col = "#B2182B", lwd = 2.5, lty = 2)
abline(h = 0, col = "gray60", lty = 3)
legend("topright", legend = c(expression(hat(f)(age)), "Partial residuals"),
       col = c("#B2182B", "black"), lty = c(2, NA), pch = c(NA, 16),
       lwd = c(2.5, NA), bty = "n", cex = 0.8)

dev.off()

# ── 5. Decision boundary in (age, glucose) plane ─────────────

glu_idx  <- which(colnames(X) == "glucose")
x_bar    <- colMeans(X)
age_grid <- seq(min(age), max(age), length.out = 150)
t_grid   <- (age_grid - min(age)) / (max(age) - min(age))
glu_grid <- seq(min(X[,glu_idx]), max(X[,glu_idx]), length.out = 150)

# Interpolate f_hat on t_grid
f_interp_rte <- approx(t_vec[ord], fit_rte$f_hat[ord], xout = t_grid, rule = 2)$y
f_interp_lse <- approx(t_vec[ord], fit_lse$f_hat[ord], xout = t_grid, rule = 2)$y

# Boundary: glucose = -[sum_{j != glucose} bar_x_j * β_j + f(t)] / β_glucose
off_rte <- sum(x_bar[-glu_idx] * fit_rte$beta[-glu_idx])
off_lse <- sum(x_bar[-glu_idx] * fit_lse$beta[-glu_idx])
off_pr  <- sum(x_bar[-glu_idx] * fit_pr$beta[-glu_idx])

db_rte <- -(off_rte + f_interp_rte) / fit_rte$beta[glu_idx]
db_lse <- -(off_lse + f_interp_lse) / fit_lse$beta[glu_idx]
db_pr  <- rep(-off_pr / fit_pr$beta[glu_idx], length(t_grid))

# Convert boundary from standardized scale back to original
glu_mean <- mean(X_raw[,glu_idx]); glu_sd <- sd(X_raw[,glu_idx])
db_rte_orig <- db_rte * glu_sd + glu_mean
db_lse_orig <- db_lse * glu_sd + glu_mean
db_pr_orig  <- db_pr  * glu_sd + glu_mean

# Probability field for IWSRTE
prob_grid <- matrix(NA, length(t_grid), length(glu_grid))
for (i in seq_along(t_grid)) {
  for (j in seq_along(glu_grid)) {
    x_new <- x_bar; x_new[glu_idx] <- glu_grid[j]
    eta <- sum(x_new * fit_rte$beta) + f_interp_rte[i]
    prob_grid[i, j] <- expit(eta)
  }
}
glu_grid_orig <- glu_grid * glu_sd + glu_mean

png("figures_real/fig_decision_real.png", width = 850, height = 600, res = 120)
par(mar = c(4.5, 4.5, 3, 1))

image(age_grid, glu_grid_orig, prob_grid,
      col = colorRampPalette(c("#2166AC","#F7F7F7","#B2182B"))(100),
      zlim = c(0, 1),
      xlab = "Age (years)", ylab = "Plasma glucose concentration",
      main = "Decision boundary in (age, glucose) plane")

# True data points
age_pts <- age
glu_pts <- X_raw[, glu_idx]
points(age_pts[y == 1], glu_pts[y == 1], pch = 16, cex = 0.6,
       col = adjustcolor("#B2182B", 0.6))
points(age_pts[y == 0], glu_pts[y == 0], pch = 4, cex = 0.6,
       col = adjustcolor("#2166AC", 0.6))

# Boundaries
yr <- range(glu_grid_orig)
in_rte <- db_rte_orig >= yr[1] & db_rte_orig <= yr[2]
lines(age_grid[in_rte], db_rte_orig[in_rte], lwd = 2.5, col = "#2166AC")
in_lse <- db_lse_orig >= yr[1] & db_lse_orig <= yr[2]
lines(age_grid[in_lse], db_lse_orig[in_lse], lwd = 2, col = "#B2182B", lty = 2)
lines(age_grid, db_pr_orig, lwd = 2, col = "#4DAF4A", lty = 4)

legend("bottomright", bg = "white", cex = 0.75,
       legend = c(sprintf("IWSRTE (k=%.2f)", best$k),
                  "IWSLSE (k=0)",
                  "ParRidge",
                  "positive (y=1)",
                  "negative (y=0)"),
       col = c("#2166AC", "#B2182B", "#4DAF4A",
               adjustcolor("#B2182B", 0.6), adjustcolor("#2166AC", 0.6)),
       lty = c(1, 2, 4, NA, NA),
       pch = c(NA, NA, NA, 16, 4),
       lwd = c(2.5, 2, 2, NA, NA))
dev.off()

cat("\nAll real data outputs saved.\n")