#!/usr/bin/env Rscript
# Monte Carlo simulation for the IWSRTE (Section 5 and Appendix A7 of the paper)
# Base R only. Outputs: all_res.rds, tables/*.csv, figures/*.png

cat("IWSRTE simulation study\n")
cat("Start:", format(Sys.time()), "\n\n")

# Functions

logit  <- function(p) log(p / (1 - p))
expit  <- function(x) 1 / (1 + exp(-x))
clip_p <- function(p, eps = 1e-8) pmin(pmax(p, eps), 1 - eps)

# Local linear smoother, Eq. (2.4) with q = 1 and Epanechnikov kernel (Fan and Gijbels, 1996)
# row i of S: e1' (T' W T)^{-1} T' W at t_i
build_smoother <- function(t_vec, h) {
  n <- length(t_vec)
  S <- matrix(0, n, n)
  for (i in 1:n) {
    u <- (t_vec - t_vec[i]) / h
    w <- ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0)
    d <- t_vec - t_vec[i]
    sw <- sum(w); swd <- sum(w*d); swd2 <- sum(w*d^2)
    TwT <- matrix(c(sw, swd, swd, swd2), 2, 2)
    if (rcond(TwT) > 1e-14) {
      Tw <- rbind(w, w*d)
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

# IWSLSE (k = 0), Eqs. (2.6)-(2.8):
# beta = solve(Xt' Om Xt) Xt' Om Vt,  Xt = (I - S) X,  Vt = (I - S) V
# f = S (V - X beta),  V = logit(pi) + (y - pi) / w,  w = max(pi (1 - pi), W_MIN)
irls_converge <- function(y, X, S_h) {
  n <- nrow(X); p <- ncol(X)
  M  <- diag(n) - S_h
  Xt <- M %*% X

  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))

  for (s in 1:max_iter) {
    b_old <- beta
    om <- pmax(pi_hat * (1 - pi_hat), W_MIN)
    V  <- logit(pi_hat) + (y - pi_hat) / om
    Vt <- as.numeric(M %*% V)

    A <- t(Xt) %*% (om * Xt)
    b <- t(Xt) %*% (om * Vt)
    beta <- as.numeric(solve(A, b))

    f_hat  <- as.numeric(S_h %*% (V - X %*% beta))
    eta    <- as.numeric(X %*% beta + f_hat)
    pi_hat <- clip_p(expit(eta))

    if (sqrt(sum((beta - b_old)^2)) < tol) break
  }
  conv <- sqrt(sum((beta - b_old)^2)) < tol

  list(beta0=beta, f0=f_hat, pi0=pi_hat,
       om=om, V=V, Vt=Vt, Xt=Xt, M=M, S_h=S_h, X=X,
       A=A, b_vec=b, conv=conv,
       ok=all(is.finite(beta)) && max(abs(beta)) < 1e6 && conv)
}

# Main method: IWSRTE at fixed (h, k), Eqs. (2.10)-(2.12) and (3.1)
# beta_hat = solve(R_k) Xt' Om Vt,  R_k = Xt' Om Xt + k I
# f_hat = S (V - X beta_hat)
# hat matrix P = S + Xt solve(R_k) Xt' Om (I - S)
# sandwich variance: solve(R_k) Xt' A Xt solve(R_k),  A = Om (I - S) Om^{-1} (I - S)' Om
ridge_at_k <- function(y, X, S_h, k) {
  n <- nrow(X); p <- ncol(X)
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
  Rk   <- XtOX + k * diag(p)
  # Important: t(om * Xt) = Xt' Om
  P_hk <- S_h + Xt %*% (solve(Rk, t(om * Xt)) %*% M)
  Sig_info <- solve(Rk, XtOX) %*% solve(Rk)    # information-based form, only for comparison
  # Xt' A Xt = C' Om^{-1} C with C = (I - S)' Om Xt
  C    <- crossprod(M, om * Xt)
  XtAX <- t(C) %*% (C / om)
  Sig  <- solve(Rk, XtAX) %*% solve(Rk)
  se      <- sqrt(pmax(diag(Sig), 0))
  se_info <- sqrt(pmax(diag(Sig_info), 0))
  d_min   <- min(eigen(XtOX, symmetric = TRUE, only.values = TRUE)$values)

  list(beta=beta, f_hat=f_hat, pi_hat=pi_hat, P_hk=P_hk, se=se, se_info=se_info,
       V=V, om=om, d_min=d_min, conv=conv,
       ok=all(is.finite(beta)) && max(abs(beta)) < 1e6)
}

# Deviance-based criteria, Eqs. (4.8)-(4.10)
# D = -2 sum{y log(pi) + (1 - y) log(1 - pi)},  df = tr(P)
sel_crit <- function(y, P_hk, pi_hat) {
  n    <- length(y)
  tr_P <- sum(diag(P_hk))
  pc   <- clip_p(pi_hat)
  D    <- -2 * sum(y * log(pc) + (1 - y) * log(1 - pc))
  c(GCV  = n * D / max(n - tr_P, 1)^2,
    AICc = D + 2 * tr_P + 2 * tr_P * (tr_P + 1) / max(n - tr_P - 1, 1),
    BIC  = D + log(n) * tr_P)
}

# Par-Ridge competitor: logistic ridge without f, IRLS with solve(X' Om X + k I) X' Om V
fit_par_ridge <- function(y, X, k=0) {
  n <- nrow(X); p <- ncol(X)
  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))

  for (s in 1:max_iter) {
    b_old <- beta
    om <- pmax(pi_hat * (1 - pi_hat), W_MIN)
    V  <- logit(pi_hat) + (y - pi_hat) / om
    beta <- as.numeric(solve(t(X)%*%(om*X) + k*diag(p), t(X)%*%(om*V)))
    pi_hat <- clip_p(expit(as.numeric(X %*% beta)))
    if (sqrt(sum((beta - b_old)^2)) < tol) break
  }
  list(beta=beta, pi_hat=pi_hat)
}

# Par-Ridge hat matrix X solve(X' Om X + k I) X' Om
par_ridge_hat <- function(X, pi_hat, k) {
  om_pr <- pmax(pi_hat * (1 - pi_hat), W_MIN)
  X %*% solve(t(X) %*% (om_pr * X) + k * diag(ncol(X)), t(om_pr * X))
}

# Data generation, Section 5.1: x_ij = sqrt(1 - rho^2) z_ij + rho z_i0, t ~ U(0, 1)
gen_data <- function(n, p, rho, beta) {
  z0 <- rnorm(n)
  Z  <- matrix(rnorm(n*p), n, p)
  X  <- sqrt(1 - rho^2) * Z + rho * z0
  tv <- runif(n)
  ft <- f_true(tv)
  eta <- as.numeric(X %*% beta + ft)
  y <- rbinom(n, 1, expit(eta))
  list(y=y, X=X, t=tv, ft=ft)
}

# Parameters

R_rep  <- 500
h_grid <- seq(0.08, 0.20, by = 0.05)       # 0.08, 0.13, 0.18
k_grid <- c(0, seq(0.05, 2.0, by = 0.10),
            2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 50)
# Important: k = 0 is not eligible in selection; it is used only for the IWSLSE end of the k-path
TRIM      <- 0.05      # trimmed-mean fraction for SMSE
EDGE      <- 0.10      # boundary region t < EDGE or t > 1 - EDGE
MAIN_CRIT <- "BIC"     # criterion for the main results
W_MIN     <- 1e-3      # floor on working weights

n_vec   <- c(100, 250, 400)
rho_vec <- c(0.90, 0.99, 0.999)
p_vec   <- c(3, 6)

beta_list <- list(
  "3" = c(1, 2, -1),
  "6" = c(1, 2, -1, 1.5, -0.5, 0.8)
)

f_true   <- function(t) sin(2 * pi * t) + 0.5 * sin(6 * pi * t)
max_iter <- 200
tol      <- 1e-6

dir.create("figures", showWarnings = FALSE)
dir.create("tables",  showWarnings = FALSE)
set.seed(2025)

# Simulation

all_res <- list()
cid <- 0

for (p in p_vec) {
  bt <- beta_list[[as.character(p)]]
  for (n in n_vec) {
    for (rho in rho_vec) {
      cid <- cid + 1
      tag <- sprintf("p%d_n%d_rho%s", p, n,
                     gsub("\\.", "", sprintf("%.3f", rho)))
      cat(sprintf("[%02d] p=%d n=%3d rho=%.3f ", cid, p, n, rho))

      B_rte  <- matrix(NA, R_rep, p)
      B_lse  <- matrix(NA, R_rep, p)
      B_pr   <- matrix(NA, R_rep, p)
      SE_rte <- matrix(NA, R_rep, p)
      KP_smse <- matrix(NA, R_rep, length(k_grid))
      KP_beta <- array(NA, dim=c(R_rep, length(k_grid), p))
      Fsup_rte <- Fsup_lse <- MCR_rte <- MCR_lse <- MCR_pr <- rep(NA, R_rep)
      CR_smse <- matrix(NA, R_rep, 3, dimnames=list(NULL,c("GCV","AICc","BIC")))
      SEi_rte <- matrix(NA, R_rep, p)
      B_ker   <- matrix(NA, R_rep, p)
      Khat <- Hhat <- Dmin <- Khat_ker <- Hhat_ker <- rep(NA, R_rep)
      Conv_rte <- rep(NA, R_rep)
      Fsup_ker <- MCR_ker <- Fedge_rte <- Fedge_lse <- Fedge_ker <- rep(NA, R_rep)

      for (r in 1:R_rep) {
        # Data generation
        d <- gen_data(n, p, rho, bt)
        sms <- lapply(h_grid, function(h) build_smoother(d$t, h))
        edge <- d$t < EDGE | d$t > 1 - EDGE

        # IWSRTE: (h, k) selection, only converged fits are eligible
        best <- list(
          MAIN = list(v=Inf, beta=NA, pi=NA, f=NA, ih=NA, se=NA),
          GCV  = list(v=Inf, beta=NA),
          AICc = list(v=Inf, beta=NA),
          BIC  = list(v=Inf, beta=NA))

        for (ih in seq_along(h_grid)) {
          for (ik in seq_along(k_grid)) {
            if (k_grid[ik] == 0) next
            rk <- tryCatch(ridge_at_k(d$y, d$X, sms[[ih]], k_grid[ik]),
                           error=function(e) NULL)
            if (is.null(rk) || !rk$ok || !rk$conv) next
            cr <- tryCatch(sel_crit(d$y, rk$P_hk, rk$pi_hat),
                           error=function(e) c(GCV=Inf,AICc=Inf,BIC=Inf))

            if (is.finite(cr["GCV"]) && cr["GCV"] < best$GCV$v)
              best$GCV <- list(v=cr["GCV"], beta=rk$beta)
            if (is.finite(cr[MAIN_CRIT]) && cr[MAIN_CRIT] < best$MAIN$v)
              best$MAIN <- list(v=cr[MAIN_CRIT], beta=rk$beta, pi=rk$pi_hat,
                               f=rk$f_hat, ih=ih, se=rk$se,
                               se_info=rk$se_info, k=k_grid[ik],
                               d_min=rk$d_min, conv=rk$conv)
            if (is.finite(cr["AICc"]) && cr["AICc"] < best$AICc$v)
              best$AICc <- list(v=cr["AICc"], beta=rk$beta)
            if (is.finite(cr["BIC"]) && cr["BIC"] < best$BIC$v)
              best$BIC <- list(v=cr["BIC"], beta=rk$beta)
          }
        }

        # IWSLSE and k-path (Figure 1) at the selected h
        if (!is.na(best$MAIN$ih)) {
          cf2 <- tryCatch(irls_converge(d$y, d$X, sms[[best$MAIN$ih]]),
                          error=function(e) NULL)
          if (!is.null(cf2) && cf2$ok) {
            for (ik in seq_along(k_grid)) {
              rk2 <- tryCatch(ridge_at_k(d$y, d$X, sms[[best$MAIN$ih]], k_grid[ik]), error=function(e) NULL)
              if (!is.null(rk2) && rk2$conv) {
                KP_beta[r,ik,] <- rk2$beta
                KP_smse[r,ik]  <- sum((rk2$beta - bt)^2)
              }
            }
            B_lse[r,]  <- cf2$beta0
            Fsup_lse[r] <- max(abs(cf2$f0 - d$ft))
            Fedge_lse[r] <- max(abs(cf2$f0 - d$ft)[edge])
            MCR_lse[r]  <- mean(as.integer(cf2$pi0>0.5) != d$y)
          }
        }

        if (!all(is.na(best$MAIN$beta))) {
          B_rte[r,]   <- best$MAIN$beta
          SE_rte[r,]  <- best$MAIN$se
          Fsup_rte[r] <- max(abs(best$MAIN$f - d$ft))
          MCR_rte[r]  <- mean(as.integer(best$MAIN$pi>0.5) != d$y)
          SEi_rte[r,] <- best$MAIN$se_info
          Khat[r] <- best$MAIN$k; Hhat[r] <- h_grid[best$MAIN$ih]
          Dmin[r] <- best$MAIN$d_min; Conv_rte[r] <- best$MAIN$conv
          Fedge_rte[r] <- max(abs(best$MAIN$f - d$ft)[edge])
        }

        # Par-Ridge, k selected by MAIN_CRIT
        pr_best_gcv <- Inf
        for (k in k_grid) {
          if (k == 0) next
          pr <- tryCatch(fit_par_ridge(d$y, d$X, k), error = function(e) NULL)
          if (is.null(pr)) next
          H_pr   <- par_ridge_hat(d$X, pr$pi_hat, k)
          gcv_pr <- sel_crit(d$y, H_pr, pr$pi_hat)[MAIN_CRIT]
          if (is.finite(gcv_pr) && gcv_pr < pr_best_gcv) {
            pr_best_gcv <- gcv_pr
            B_pr[r, ] <- pr$beta
            MCR_pr[r] <- mean(as.integer(pr$pi_hat > 0.5) != d$y)
          }
        }

        # Kernel (Nadaraya-Watson) estimator, same grids and same rule
        ker_best <- Inf
        for (ih in seq_along(h_grid)) {
          S_nw <- build_nw_smoother(d$t, h_grid[ih])
          for (ik in seq_along(k_grid)) {
            if (k_grid[ik] == 0) next
            rkn <- tryCatch(ridge_at_k(d$y, d$X, S_nw, k_grid[ik]), error=function(e) NULL)
            if (is.null(rkn) || !rkn$ok || !rkn$conv) next
            crn <- tryCatch(sel_crit(d$y, rkn$P_hk, rkn$pi_hat),
                            error=function(e) c(GCV=Inf, AICc=Inf, BIC=Inf))
            if (is.finite(crn[MAIN_CRIT]) && crn[MAIN_CRIT] < ker_best) {
              ker_best <- crn[MAIN_CRIT]
              B_ker[r,]   <- rkn$beta
              MCR_ker[r]  <- mean(as.integer(rkn$pi_hat > 0.5) != d$y)
              Fsup_ker[r] <- max(abs(rkn$f_hat - d$ft))
              Fedge_ker[r] <- max(abs(rkn$f_hat - d$ft)[edge])
              Khat_ker[r] <- k_grid[ik]; Hhat_ker[r] <- h_grid[ih]
            }
          }
        }

        CR_smse[r,"GCV"]  <- if(!all(is.na(best$GCV$beta)))  sum((best$GCV$beta-bt)^2)  else NA
        CR_smse[r,"AICc"] <- if(!all(is.na(best$AICc$beta))) sum((best$AICc$beta-bt)^2) else NA
        CR_smse[r,"BIC"]  <- if(!all(is.na(best$BIC$beta)))  sum((best$BIC$beta-bt)^2)  else NA
      }

      # Metrics
      # Important: failed replications (no converged fit) are discarded, not replaced; counts are in table T9b
      ok_r <- complete.cases(B_rte)
      ok_l <- complete.cases(B_lse)
      ok_p <- complete.cases(B_pr)
      ok_ker <- complete.cases(B_ker)

      # SMSE = Bias^2 + Var, computed on the same (trimmed) replications
      agg <- function(B, ok, trim = TRIM) {
        if (sum(ok) <= 1) return(c(smse=NA, bias2=NA, var=NA))
        Bk <- B[ok, , drop=FALSE]
        se <- rowSums(sweep(Bk, 2, bt)^2)
        if (trim > 0) {
          keep <- se <= quantile(se, 1 - trim)
          Bk <- Bk[keep, , drop=FALSE]; se <- se[keep]
        }
        smse  <- mean(se)
        bias2 <- sum((colMeans(Bk) - bt)^2)
        c(smse=smse, bias2=bias2, var=smse - bias2)
      }
      a_r <- agg(B_rte, ok_r); a_l <- agg(B_lse, ok_l); a_p <- agg(B_pr, ok_p)
      smse_r <- unname(a_r["smse"]); b2_r <- unname(a_r["bias2"]); var_r <- unname(a_r["var"])
      smse_l <- unname(a_l["smse"]); b2_l <- unname(a_l["bias2"]); var_l <- unname(a_l["var"])
      smse_p <- unname(a_p["smse"])
      smse_k <- unname(agg(B_ker, ok_ker)["smse"])
      smse_raw <- c(IWSRTE=unname(agg(B_rte, ok_r, 0)["smse"]), IWSLSE=unname(agg(B_lse, ok_l, 0)["smse"]),
                    ParRidge=unname(agg(B_pr, ok_p, 0)["smse"]), KerIWSRE=unname(agg(B_ker, ok_ker, 0)["smse"]))

      # k-path decomposition (Figure 1)
      kp_m <- colMeans(KP_smse, na.rm=TRUE)
      kp_b2 <- kp_v <- rep(NA, length(k_grid))
      for (ik in seq_along(k_grid)) {
        bk <- KP_beta[,ik,]; ok_k <- complete.cases(bk)
        if (sum(ok_k)>1) {
          kp_b2[ik] <- sum((colMeans(bk[ok_k,,drop=F])-bt)^2)
          kp_v[ik]  <- kp_m[ik] - kp_b2[ik]
        }
      }

      # Coverage
      # cv: uncorrected Wald, beta_hat +- 1.96 SE with SE from Eq. (3.1) (Table 3)
      # cv_info: same with the information-based SE; cv_mc: centred at the Monte Carlo mean (check only)
      cv <- cv_info <- cv_mc <- se_sd <- rep(NA, p)
      if (sum(ok_r) > 5) {
        bhat <- colMeans(B_rte[ok_r,,drop=FALSE]) - bt
        for (j in 1:p) {
          bj <- B_rte[ok_r,j]; sj <- SE_rte[ok_r,j]; si <- SEi_rte[ok_r,j]
          cv[j]      <- mean(abs(bj - bt[j]) <= 1.96*sj, na.rm=TRUE)
          cv_info[j] <- mean(abs(bj - bt[j]) <= 1.96*si, na.rm=TRUE)
          cv_mc[j]   <- mean(abs(bj - bhat[j] - bt[j]) <= 1.96*sj, na.rm=TRUE)
          se_sd[j]   <- mean(sj, na.rm=TRUE) / sd(bj)
        }
      }

      # Replication counts
      reps <- c(requested=R_rep, valid_RTE=sum(ok_r), valid_LSE=sum(ok_l),
                valid_PR=sum(ok_p), valid_KER=sum(ok_ker),
                nonconv_RTE=sum(Conv_rte == FALSE, na.rm=TRUE),
                no_fit_RTE=sum(!ok_r), no_fit_LSE=sum(!ok_l))

      cat(sprintf("SMSE: R=%.4f L=%.4f P=%.4f K=%.4f | valid R=%d L=%d P=%d K=%d | nonconv R=%d\n",
                  smse_r, smse_l, smse_p, smse_k, sum(ok_r), sum(ok_l), sum(ok_p), sum(ok_ker),
                  reps["nonconv_RTE"]))

      # Monte Carlo standard errors (untrimmed) and paired z = mean(d) / (sd(d) / sqrt(R'))
      sq_err <- function(B) rowSums(sweep(B, 2, bt)^2)
      mc_one <- function(B, ok) { e <- sq_err(B[ok,,drop=FALSE]); c(smse=mean(e), mcse=sd(e)/sqrt(length(e))) }
      mc_pair <- function(B2, ok2) {
        cm <- ok_r & ok2
        dd <- sq_err(B_rte[cm,,drop=FALSE]) - sq_err(B2[cm,,drop=FALSE])
        c(diff=mean(dd), mcse=sd(dd)/sqrt(sum(cm)), z=mean(dd)/(sd(dd)/sqrt(sum(cm))), n=sum(cm))
      }
      mcse <- list(one=rbind(IWSRTE=mc_one(B_rte, ok_r), IWSLSE=mc_one(B_lse, ok_l),
                             ParRidge=mc_one(B_pr, ok_p), KerIWSRE=mc_one(B_ker, ok_ker)),
                   pair=rbind(vsLSE=mc_pair(B_lse, ok_l), vsPR=mc_pair(B_pr, ok_p),
                              vsKER=mc_pair(B_ker, ok_ker)),
                   cov=sqrt(cv * (1 - cv) / sum(ok_r)),
                   se_sd_med=sapply(1:p, function(j)
                     median(SE_rte[ok_r,j], na.rm=TRUE) / (IQR(B_rte[ok_r,j]) / 1.349)))

      all_res[[tag]] <- list(
        p=p, n=n, rho=rho, tag=tag, mcse=mcse,
        smse=c(IWSRTE=smse_r, IWSLSE=smse_l, ParRidge=smse_p, KerIWSRE=smse_k),
        smse_raw=smse_raw, reps=reps,
        cov_info=cv_info, cov_mc=cv_mc, se_sd=se_sd,
        sel=c(k_med=median(Khat,na.rm=T), h_med=median(Hhat,na.rm=T),
              k0_share=mean(Khat==0,na.rm=T), dmin_med=median(Dmin,na.rm=T),
              k_med_ker=median(Khat_ker,na.rm=T), h_med_ker=median(Hhat_ker,na.rm=T)),
        fedge=c(IWSRTE=median(Fedge_rte,na.rm=T), IWSLSE=median(Fedge_lse,na.rm=T),
                KerIWSRE=median(Fedge_ker,na.rm=T)),
        bias2=c(IWSRTE=b2_r, IWSLSE=b2_l),
        var=c(IWSRTE=var_r, IWSLSE=var_l),
        kp_k=k_grid, kp_smse=kp_m, kp_bias2=kp_b2, kp_var=kp_v,
        cov=cv,
        fsup=c(IWSRTE=median(Fsup_rte,na.rm=T), IWSLSE=median(Fsup_lse,na.rm=T),
               KerIWSRE=median(Fsup_ker,na.rm=T)),
        mcr=c(IWSRTE=mean(MCR_rte,na.rm=T), IWSLSE=mean(MCR_lse,na.rm=T),
              ParRidge=mean(MCR_pr,na.rm=T), KerIWSRE=mean(MCR_ker,na.rm=T)),
        crit_smse=apply(CR_smse, 2, function(x){
          x <- x[is.finite(x)]
          if (TRIM > 0 && length(x) > 1) x <- x[x <= quantile(x, 1 - TRIM)]
          mean(x)
        }),
        B_rte=B_rte, ok_rte=ok_r, B_ker=B_ker, ok_ker=ok_ker, Khat=Khat,
        B_lse=B_lse, ok_lse=ok_l, B_pr=B_pr, ok_pr=ok_p, SE_rte=SE_rte)
    }
  }
}

saveRDS(all_res, "all_res.rds")
cat("\nSaved all_res.rds\n")

# Tables (T = table number in the paper)

t1 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             IWSRTE=round(r$smse["IWSRTE"],6),
             IWSLSE=round(r$smse["IWSLSE"],6),
             ParRidge=round(r$smse["ParRidge"],6),
             KerIWSRE=round(r$smse["KerIWSRE"],6))))
rownames(t1)<-NULL; write.csv(t1,"tables/T1_smse.csv",row.names=F)

t2 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             Bias2_RTE=round(r$bias2["IWSRTE"],6),Var_RTE=round(r$var["IWSRTE"],6),
             SMSE_RTE=round(r$smse["IWSRTE"],6),
             Bias2_LSE=round(r$bias2["IWSLSE"],6),Var_LSE=round(r$var["IWSLSE"],6),
             SMSE_LSE=round(r$smse["IWSLSE"],6))))
rownames(t2)<-NULL; write.csv(t2,"tables/T2_bias_variance.csv",row.names=F)

t3 <- do.call(rbind, lapply(all_res, function(r) {
  df <- data.frame(p=r$p, n=r$n, rho=r$rho)
  for(j in seq_along(r$cov)) df[[paste0("beta",j)]] <- round(r$cov[j],3)
  if(length(r$cov) < max(p_vec)) {
    for(j in (length(r$cov)+1):max(p_vec))
      df[[paste0("beta",j)]] <- NA
  }
  df
}))
rownames(t3)<-NULL; write.csv(t3,"tables/T3_coverage.csv",row.names=F)

t4 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             IWSRTE=round(r$mcr["IWSRTE"],4),
             IWSLSE=round(r$mcr["IWSLSE"],4),
             ParRidge=round(r$mcr["ParRidge"],4),
             KerIWSRE=round(r$mcr["KerIWSRE"],4))))
rownames(t4)<-NULL; write.csv(t4,"tables/T6_mcr.csv",row.names=F)

t5 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             GCV=round(r$crit_smse["GCV"],6),
             AICc=round(r$crit_smse["AICc"],6),
             BIC=round(r$crit_smse["BIC"],6))))
rownames(t5)<-NULL; write.csv(t5,"tables/T7_criteria.csv",row.names=F)

pad <- function(x) c(round(x,3), rep(NA, max(p_vec) - length(x)))
t6 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             k_med=r$sel["k_med"], k0_share=round(r$sel["k0_share"],3),
             h_med=r$sel["h_med"], dmin_med=signif(r$sel["dmin_med"],4),
             k_med_ker=r$sel["k_med_ker"], h_med_ker=r$sel["h_med_ker"])))
rownames(t6)<-NULL; write.csv(t6,"tables/T9a_selected_hk.csv",row.names=F)

t7 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho, t(r$reps))))
rownames(t7)<-NULL; write.csv(t7,"tables/T9b_replications.csv",row.names=F)

t8 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             sup_RTE=round(r$fsup["IWSRTE"],4), sup_LSE=round(r$fsup["IWSLSE"],4),
             sup_KER=round(r$fsup["KerIWSRE"],4),
             edge_RTE=round(r$fedge["IWSRTE"],4), edge_LSE=round(r$fedge["IWSLSE"],4),
             edge_KER=round(r$fedge["KerIWSRE"],4))))
rownames(t8)<-NULL; write.csv(t8,"tables/T10_fhat_error.csv",row.names=F)

# Coverage checks and SE/SD ratio (Section 5.3 text)
t9 <- do.call(rbind, lapply(all_res, function(r) {
  df <- data.frame(p=r$p,n=r$n,rho=r$rho)
  cvi <- pad(r$cov_info); cvm <- pad(r$cov_mc); ssd <- pad(r$se_sd)
  for (j in 1:max(p_vec)) {
    df[[paste0("covInfo_b",j)]] <- cvi[j]
    df[[paste0("covMC_b",j)]]   <- cvm[j]
    df[[paste0("SEoverSD_b",j)]] <- ssd[j]
  }
  df
}))
rownames(t9)<-NULL; write.csv(t9,"tables/text_coverage_checks.csv",row.names=F)

t10 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho, t(round(r$smse_raw,6)))))
rownames(t10)<-NULL; write.csv(t10,"tables/text_smse_untrimmed.csv",row.names=F)

t11 <- do.call(rbind, lapply(all_res, function(r) {
  m <- r$mcse
  data.frame(p=r$p, n=r$n, rho=r$rho,
             SMSE_RTE=round(m$one["IWSRTE","smse"],4), MCSE_RTE=round(m$one["IWSRTE","mcse"],4),
             SMSE_LSE=round(m$one["IWSLSE","smse"],4), MCSE_LSE=round(m$one["IWSLSE","mcse"],4),
             SMSE_PR=round(m$one["ParRidge","smse"],4), MCSE_PR=round(m$one["ParRidge","mcse"],4),
             SMSE_KER=round(m$one["KerIWSRE","smse"],4), MCSE_KER=round(m$one["KerIWSRE","mcse"],4),
             d_LSE=round(m$pair["vsLSE","diff"],4), z_LSE=round(m$pair["vsLSE","z"],2),
             d_PR=round(m$pair["vsPR","diff"],4),   z_PR=round(m$pair["vsPR","z"],2),
             d_KER=round(m$pair["vsKER","diff"],4), z_KER=round(m$pair["vsKER","z"],2),
             covMCSE_max=if (all(is.na(m$cov))) NA else round(max(m$cov, na.rm=TRUE),4),
             SEoverSD_med_min=round(min(m$se_sd_med),3), SEoverSD_med_max=round(max(m$se_sd_med),3))
}))
rownames(t11)<-NULL; write.csv(t11,"tables/T8_mcse.csv",row.names=F)
cat("Tables saved.\n")

cat("\nTable 1 (trimmed SMSE)\n"); print(t1)
cat("\nTable 3 (uncorrected Wald coverage)\n"); print(t3)
cat("\nSelected h, k\n"); print(t6)
cat("\nReplications (requested, valid, failed)\n"); print(t7)
cat("\nf-hat sup error (all t / boundary)\n"); print(t8)
cat("\nMCSE and paired differences (IWSRTE minus other; |z| > 2 means a clear difference)\n"); print(t11)

# Consistency checks
chk1 <- max(abs(t1$IWSRTE - t5[[MAIN_CRIT]]), na.rm=TRUE)
chk2 <- max(abs(t2$SMSE_RTE - (t2$Bias2_RTE + t2$Var_RTE)), na.rm=TRUE)
chk3 <- max(abs(t2$SMSE_LSE - (t2$Bias2_LSE + t2$Var_LSE)), na.rm=TRUE)
chk4 <- min(t2$SMSE_RTE - t2$Bias2_RTE, na.rm=TRUE)
cat(sprintf("\n[CHECK] |Table1.IWSRTE - Table7.%s| max = %.3e (want ~0)\n", MAIN_CRIT, chk1))
cat(sprintf("[CHECK] |SMSE - (Bias2+Var)| RTE max = %.3e, LSE max = %.3e (want ~0)\n", chk2, chk3))
cat(sprintf("[CHECK] min(Var_RTE) = %.3e (want >= 0)\n", chk4))

# Figures
C1<-"#2166AC"; C2<-"#B2182B"; C3<-"#4DAF4A"

# Figure 1: k-path
for(tag in names(all_res)) {
  r<-all_res[[tag]]; if(all(is.na(r$kp_smse))) next
  png(sprintf("figures/kpath_%s.png",tag),width=700,height=500,res=120)
  par(mar=c(4.5,4.5,3,1))
  ym<-max(c(r$kp_smse,r$kp_bias2,r$kp_var),na.rm=T)*1.15
  plot(r$kp_k,r$kp_smse,type="l",lwd=2.5,col=C1,ylim=c(0,max(ym,0.01)),
       xlab="Ridge parameter k",ylab="Value",
       main=bquote(p==.(r$p)~", "~n==.(r$n)~", "~rho==.(r$rho)))
  lines(r$kp_k,r$kp_bias2,lwd=2,col=C2,lty=2)
  lines(r$kp_k,r$kp_var,lwd=2,col=C3,lty=3)
  ki<-which.min(r$kp_smse)
  points(r$kp_k[ki],r$kp_smse[ki],pch=19,col=C1,cex=1.4)
  legend("topright",legend=c("SMSE",expression(Bias^2),"Variance",expression(k^"*")),
         col=c(C1,C2,C3,C1),lty=c(1,2,3,NA),pch=c(NA,NA,NA,19),
         lwd=c(2.5,2,2,NA),bty="n",cex=0.8)
  dev.off()
}

# Figure 4: QQ plots
for(tag in names(all_res)) {
  r<-all_res[[tag]]; if(sum(r$ok_rte)<8) next
  bt1<-beta_list[[as.character(r$p)]][1]
  z<-sqrt(r$n)*(r$B_rte[r$ok_rte,1]-bt1)
  png(sprintf("figures/qq_%s.png",tag),width=550,height=500,res=120)
  par(mar=c(4.5,4.5,3,1))
  qqnorm(z,main=bquote(sqrt(n)(hat(beta)[1]-beta[1])~": "~
                         p==.(r$p)~", "~n==.(r$n)~", "~rho==.(r$rho)),pch=16,cex=0.7,col=C1)
  qqline(z,col=C2,lwd=2)
  dev.off()
}

# Figure 5: f-hat convergence
for(p in p_vec) { for(rho in rho_vec) {
  rtag<-gsub("\\.","",sprintf("%.3f",rho))
  ns<-c(); fs<-c()
  for(n in n_vec) {
    tg<-sprintf("p%d_n%d_rho%s",p,n,rtag)
    if(!is.null(all_res[[tg]])){ns<-c(ns,n);fs<-c(fs,all_res[[tg]]$fsup["IWSRTE"])}
  }
  if(length(ns)<2||any(is.na(fs))||any(fs<=0)) next
  png(sprintf("figures/fconv_p%d_rho%s.png",p,rtag),width=600,height=480,res=120)
  par(mar=c(4.5,4.5,3,1))
  plot(log(ns),log(fs),type="b",pch=19,lwd=2,col=C1,
       xlab="log(n)",ylab=expression(log(max~"|"~hat(f)-f~"|")),
       main=bquote(hat(f)~"convergence: "~p==.(p)~", "~rho==.(rho)))
  abline(a=log(fs[1])+(1/3)*log(ns[1]),b=-1/3,col="gray50",lty=2,lwd=1.5)
  legend("topright",legend=c("Empirical",expression("Slope"~-1/3)),
         col=c(C1,"gray50"),lty=c(1,2),pch=c(19,NA),lwd=c(2,1.5),bty="n",cex=0.85)
  dev.off()
}}

# Figure 2: fitted f-hat vs true f(t), one data set per setting at h = median(h_grid)
for(tag in names(all_res)) {
  r <- all_res[[tag]]
  bt <- beta_list[[as.character(r$p)]]
  set.seed(2025 + which(names(all_res) == tag))
  d <- gen_data(r$n, r$p, r$rho, bt)
  h_mid <- median(h_grid)
  S_h <- build_smoother(d$t, h_mid)
  best_gcv_val <- Inf; best_k <- 0
  for (k_try in k_grid) {
    if (k_try == 0) next
    rk_try <- tryCatch(ridge_at_k(d$y, d$X, S_h, k_try), error=function(e) NULL)
    if (is.null(rk_try) || !rk_try$ok || !rk_try$conv) next
    cr_try <- tryCatch(sel_crit(d$y, rk_try$P_hk, rk_try$pi_hat), error=function(e) c(GCV=Inf, AICc=Inf, BIC=Inf))
    if (is.finite(cr_try[MAIN_CRIT]) && cr_try[MAIN_CRIT] < best_gcv_val) {
      best_gcv_val <- cr_try[MAIN_CRIT]; best_k <- k_try
    }
  }
  rk0 <- tryCatch(ridge_at_k(d$y, d$X, S_h, 0),      error=function(e) NULL)
  rk2 <- tryCatch(ridge_at_k(d$y, d$X, S_h, best_k), error=function(e) NULL)
  if (is.null(rk0) || is.null(rk2)) next
  ord <- order(d$t)
  png(sprintf("figures/fhat_%s.png", tag), width=700, height=500, res=120)
  par(mar=c(4.5, 4.5, 3, 1))
  ylims <- range(c(f_true(d$t), rk0$f_hat, rk2$f_hat), na.rm=TRUE)
  ylims <- ylims + c(-0.15, 0.15) * diff(ylims)
  plot(d$t[ord], f_true(d$t[ord]), type="l", lwd=2.5, col="black",
       xlab="t", ylab="f(t)", ylim=ylims,
       main=bquote(p==.(r$p)~", "~n==.(r$n)~", "~rho==.(r$rho)))
  lines(d$t[ord], rk0$f_hat[ord], lwd=2, col=C2, lty=2)
  lines(d$t[ord], rk2$f_hat[ord], lwd=2, col=C1, lty=1)
  rug(d$t, col="gray70", ticksize=0.02)
  legend("topright",
         legend=c(expression("True " * f(t)), "IWSLSE (k = 0)",
                  bquote("IWSRTE (k = "*.(round(best_k,2))*")")),
         col=c("black", C2, C1), lty=c(1, 2, 1), lwd=c(2.5, 2, 2), bty="n", cex=0.8)
  dev.off()
}

# Figure 3: decision boundaries, same data sets as Figure 2
for(tag in names(all_res)) {
  r <- all_res[[tag]]
  bt <- beta_list[[as.character(r$p)]]
  set.seed(2025 + which(names(all_res) == tag))
  d <- gen_data(r$n, r$p, r$rho, bt)
  h_mid <- median(h_grid)
  S_h <- build_smoother(d$t, h_mid)
  best_gcv_val <- Inf; best_k <- 0
  for (k_try in k_grid) {
    if (k_try == 0) next
    rk_try <- tryCatch(ridge_at_k(d$y, d$X, S_h, k_try), error=function(e) NULL)
    if (is.null(rk_try) || !rk_try$ok || !rk_try$conv) next
    cr_try <- tryCatch(sel_crit(d$y, rk_try$P_hk, rk_try$pi_hat), error=function(e) c(GCV=Inf, AICc=Inf, BIC=Inf))
    if (is.finite(cr_try[MAIN_CRIT]) && cr_try[MAIN_CRIT] < best_gcv_val) {
      best_gcv_val <- cr_try[MAIN_CRIT]; best_k <- k_try
    }
  }
  fit_rte <- tryCatch(ridge_at_k(d$y, d$X, S_h, best_k), error=function(e) NULL)
  fit_lse <- tryCatch(ridge_at_k(d$y, d$X, S_h, 0),      error=function(e) NULL)
  pr_best_gcv <- Inf; pr_best_k <- 0
  for (k_try in k_grid) {
    if (k_try == 0) next
    pr_try <- tryCatch(fit_par_ridge(d$y, d$X, k_try), error=function(e) NULL)
    if (is.null(pr_try)) next
    H_pr   <- par_ridge_hat(d$X, pr_try$pi_hat, k_try)
    gcv_pr <- sel_crit(d$y, H_pr, pr_try$pi_hat)[MAIN_CRIT]
    if (is.finite(gcv_pr) && gcv_pr < pr_best_gcv) {
      pr_best_gcv <- gcv_pr; pr_best_k <- k_try
    }
  }
  fit_pr <- tryCatch(fit_par_ridge(d$y, d$X, pr_best_k), error=function(e) NULL)
  if (is.null(fit_rte) || is.null(fit_lse) || is.null(fit_pr)) next

  # Boundary in (t, x1) with the other covariates at their means: x1 = -(offset + f(t)) / beta_1
  ng <- 150
  t_grid  <- seq(0, 1, length.out=ng)
  x1_range <- range(d$X[,1]) * 1.2
  x1_grid <- seq(x1_range[1], x1_range[2], length.out=ng)
  x_bar <- colMeans(d$X)
  offset_true <- sum(x_bar[-1] * bt[-1])
  db_true <- -(offset_true + f_true(t_grid)) / bt[1]
  f_interp_rte <- approx(d$t[order(d$t)], fit_rte$f_hat[order(d$t)], xout=t_grid, rule=2)$y
  offset_rte <- sum(x_bar[-1] * fit_rte$beta[-1])
  db_rte <- -(offset_rte + f_interp_rte) / fit_rte$beta[1]
  f_interp_lse <- approx(d$t[order(d$t)], fit_lse$f_hat[order(d$t)], xout=t_grid, rule=2)$y
  offset_lse <- sum(x_bar[-1] * fit_lse$beta[-1])
  db_lse <- -(offset_lse + f_interp_lse) / fit_lse$beta[1]
  offset_pr <- sum(x_bar[-1] * fit_pr$beta[-1])
  db_pr <- rep(-offset_pr / fit_pr$beta[1], ng)
  prob_grid <- matrix(NA, ng, ng)
  for (i in seq_along(t_grid)) for (j in seq_along(x1_grid)) {
    x_new <- x_bar; x_new[1] <- x1_grid[j]
    prob_grid[i, j] <- expit(sum(x_new * fit_rte$beta) + f_interp_rte[i])
  }
  png(sprintf("figures/decision_%s.png", tag), width=800, height=600, res=120)
  par(mar=c(4.5, 4.5, 3, 5))
  image(t_grid, x1_grid, prob_grid,
        col=colorRampPalette(c("#2166AC","#F7F7F7","#B2182B"))(100),
        xlab="t (nonparametric covariate)",
        ylab=expression(x[1] ~ " (parametric covariate)"),
        main=bquote("Decision boundaries: " ~ p==.(r$p) ~ ", " ~ n==.(r$n) ~ ", " ~ rho==.(r$rho)),
        zlim=c(0, 1))
  contour(t_grid, x1_grid, prob_grid, levels=0.5, add=TRUE, drawlabels=FALSE, col=C1, lwd=1, lty=3)
  in_true <- db_true >= x1_range[1] & db_true <= x1_range[2]
  lines(t_grid[in_true], db_true[in_true], lwd=3, col="black")
  in_rte <- db_rte >= x1_range[1] & db_rte <= x1_range[2]
  lines(t_grid[in_rte], db_rte[in_rte], lwd=2.5, col=C1)
  in_lse <- db_lse >= x1_range[1] & db_lse <= x1_range[2]
  lines(t_grid[in_lse], db_lse[in_lse], lwd=2, col=C2, lty=2)
  lines(t_grid, db_pr, lwd=2, col=C3, lty=4)
  points(d$t[d$y==1], d$X[d$y==1, 1], pch=16, cex=0.7, col=adjustcolor("#B2182B", 0.6))
  points(d$t[d$y==0], d$X[d$y==0, 1], pch=4, cex=0.7, col=adjustcolor("#2166AC", 0.6))
  legend("bottomright", bg="white", cex=0.7,
         legend=c("True boundary",
                  bquote("IWSRTE (k="*.(round(best_k,2))*")"),
                  "IWSLSE (k=0)",
                  bquote("ParRidge (k="*.(round(pr_best_k,2))*")"),
                  "y = 1", "y = 0"),
         col=c("black", C1, C2, C3, adjustcolor("#B2182B",0.6), adjustcolor("#2166AC",0.6)),
         lty=c(1, 1, 2, 4, NA, NA), pch=c(NA, NA, NA, NA, 16, 4),
         lwd=c(3, 2.5, 2, 2, NA, NA))
  dev.off()
}

cat("\nDone:", format(Sys.time()), "\n")
