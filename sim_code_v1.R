#Simulation Study for Modified kernel ridge type estimators in semiparametric logistic regression models 
#(we can change the title)

#!/usr/bin/env Rscript
# ==============================================================
# Monte Carlo Simulation: IWSRTE for Semiparametric Logistic
#   Regression under Multicollinearity
# ==============================================================

cat("=== IWSRTE Simulation Study ===\n")
cat("Start:", format(Sys.time()), "\n\n")

# ── 0. CONFIGURATION ─────────────────────────────────────────

# --- Full settings (uncomment for production) ---
 R_rep  <- 1000
 h_grid <- seq(0.08, 0.20, by = 0.05)
 k_grid <- seq(0.05, 2.0, by = 0.10)

# --- Demo settings ---
#R_rep  <- 10
#h_grid <- seq(0.10, 0.45, by = 0.10)
#k_grid <- seq(0, 2.0, by = 0.40)

n_vec   <- c(100, 250, 400)
rho_vec <- c(0.90, 0.99, 0.999)
p_vec   <- c(3,6)

beta_list <- list(
  "3" = c(1, 2, -1),
  "6" = c(1, 2, -1, 1.5, -0.5, 0.8)
)

f_true <- function(t) sin(2 * pi * t) + 0.5 * sin(6 * pi * t)
max_iter <- 50
tol      <- 1e-6

dir.create("figures", showWarnings = FALSE)
dir.create("tables",  showWarnings = FALSE)
set.seed(2025)

# ── 1. HELPERS ────────────────────────────────────────────────

logit  <- function(p) log(p / (1 - p))
expit  <- function(x) 1 / (1 + exp(-x))
clip_p <- function(p, eps = 1e-8) pmin(pmax(p, eps), 1 - eps)

# ── 2. LOCAL LINEAR SMOOTHER ─────────────────────────────────

build_smoother <- function(t_vec, h) {
  n <- length(t_vec)
  S <- matrix(0, n, n)
  for (i in 1:n) {
    u <- (t_vec - t_vec[i]) / h
    w <- ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0)
    d <- t_vec - t_vec[i]
    # Avoid diag(w): T'W = [sum(w), sum(w*d); sum(w*d), sum(w*d^2)]
    sw <- sum(w); swd <- sum(w*d); swd2 <- sum(w*d^2)
    TwT <- matrix(c(sw, swd, swd, swd2), 2, 2)
    if (rcond(TwT) > 1e-14) {
      # T'W row vectors: [w_j, w_j*d_j] transposed gives 2×n
      Tw <- rbind(w, w*d)
      S[i, ] <- solve(TwT, Tw)[1, ]
    }
  }
  S
}

# ── 3. IRLS CONVERGENCE (k=0) ────────────────────────────────

irls_converge <- function(y, X, S_h) {
  n <- nrow(X); p <- ncol(X)
  M  <- diag(n) - S_h
  Xt <- M %*% X
  
  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))
  
  for (s in 1:max_iter) {
    b_old <- beta
    om <- pi_hat * (1 - pi_hat)
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
  
  list(beta0=beta, f0=f_hat, pi0=pi_hat,
       om=om, V=V, Vt=Vt, Xt=Xt, M=M, S_h=S_h, X=X,
       A=A, b_vec=b, ok=(s < max_iter))
}

# ── 4. RIDGE SWEEP ───────────────────────────────────────────

ridge_at_k <- function(y, X, S_h, k) {
  n <- nrow(X); p <- ncol(X)
  M  <- diag(n) - S_h
  Xt <- M %*% X
  
  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))
  
  for (s in 1:max_iter) {
    b_old <- beta
    om <- pi_hat * (1 - pi_hat)
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
  
  om <- pi_hat * (1 - pi_hat)
  Rk <- t(Xt) %*% (om * Xt) + k * diag(p)
  P_hk <- S_h + Xt %*% (solve(Rk, t(Xt) * om) %*% M)
  
  list(beta=beta, f_hat=f_hat, pi_hat=pi_hat, P_hk=P_hk, ok=(s < max_iter))
}

# ── 5. CRITERIA ──────────────────────────────────────────────

sel_crit <- function(y, P_hk) {
  n    <- length(y)
  r    <- y - as.numeric(P_hk %*% y)
  RSS  <- sum(r^2)
  tr_P <- sum(diag(P_hk))
  
  dg   <- max((1 - tr_P/n)^2, 1e-12)
  gcv  <- (RSS/n) / dg
  aicc <- log(max(RSS/n, 1e-300)) + 2*(tr_P+1)/max(n-tr_P-2, 1)
  bic  <- RSS/max(n-tr_P, 1) + log(n)/n * tr_P
  c(GCV=gcv, AICc=aicc, BIC=bic)
}

# ── 6. PARAMETRIC LOGISTIC RIDGE ─────────────────────────────

fit_par_ridge <- function(y, X, k=0) {
  n <- nrow(X); p <- ncol(X)
  beta   <- rep(0, p)
  pi_hat <- clip_p(rep(mean(y), n))
  
  for (s in 1:max_iter) {
    b_old <- beta
    om <- pi_hat * (1 - pi_hat)
    V  <- logit(pi_hat) + (y - pi_hat) / om
    beta <- as.numeric(solve(t(X)%*%(om*X) + k*diag(p), t(X)%*%(om*V)))
    pi_hat <- clip_p(expit(as.numeric(X %*% beta)))
    if (sqrt(sum((beta - b_old)^2)) < tol) break
  }
  list(beta=beta, pi_hat=pi_hat)
}

# ── 7. DATA GENERATION ──────────────────────────────────────

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

# ── 8. MAIN LOOP ────────────────────────────────────────────

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
      
      B_rte <- matrix(NA, R_rep, p)
      B_lse <- matrix(NA, R_rep, p)
      B_pr  <- matrix(NA, R_rep, p)
      KP_smse <- matrix(NA, R_rep, length(k_grid))
      KP_beta <- array(NA, dim=c(R_rep, length(k_grid), p))
      Fsup_rte <- Fsup_lse <- MCR_rte <- MCR_lse <- MCR_pr <- rep(NA, R_rep)
      CR_smse <- matrix(NA, R_rep, 3, dimnames=list(NULL,c("GCV","AICc","BIC")))
      
      for (r in 1:R_rep) {
        #cat("simulation", r, "Ended.", "\n")
        d <- gen_data(n, p, rho, bt)
        sms <- lapply(h_grid, function(h) build_smoother(d$t, h))
        
        best <- list(
          GCV  = list(v=Inf, beta=NA, pi=NA, f=NA, ih=NA),
          AICc = list(v=Inf, beta=NA),
          BIC  = list(v=Inf, beta=NA))
        
        for (ih in seq_along(h_grid)) {
          cf <- tryCatch(irls_converge(d$y, d$X, sms[[ih]]),
                         error=function(e) NULL)
          if (is.null(cf) || !cf$ok) next
          
          for (ik in seq_along(k_grid)) {
            rk <- tryCatch(ridge_at_k(d$y, d$X, sms[[ih]], k_grid[ik]),
                           error=function(e) NULL)
            if (is.null(rk) || !rk$ok) next
            cr <- tryCatch(sel_crit(d$y, rk$P_hk),
                           error=function(e) c(GCV=Inf,AICc=Inf,BIC=Inf))
            
            if (is.finite(cr["GCV"]) && cr["GCV"] < best$GCV$v)
              best$GCV <- list(v=cr["GCV"], beta=rk$beta, pi=rk$pi_hat,
                               f=rk$f_hat, ih=ih)
            if (is.finite(cr["AICc"]) && cr["AICc"] < best$AICc$v)
              best$AICc <- list(v=cr["AICc"], beta=rk$beta)
            if (is.finite(cr["BIC"]) && cr["BIC"] < best$BIC$v)
              best$BIC <- list(v=cr["BIC"], beta=rk$beta)
          }
        }
        
        # k-path at GCV-best h
        if (!is.na(best$GCV$ih)) {
          cf2 <- tryCatch(irls_converge(d$y, d$X, sms[[best$GCV$ih]]),
                          error=function(e) NULL)
          if (!is.null(cf2) && cf2$ok) {
            for (ik in seq_along(k_grid)) {
              rk2 <- tryCatch(ridge_at_k(d$y, d$X, sms[[best$GCV$ih]], k_grid[ik]), error=function(e) NULL)
              if (!is.null(rk2)) {
                KP_beta[r,ik,] <- rk2$beta
                KP_smse[r,ik]  <- sum((rk2$beta - bt)^2)
              }
            }
            B_lse[r,]  <- cf2$beta0
            Fsup_lse[r] <- max(abs(cf2$f0 - d$ft))
            MCR_lse[r]  <- mean(as.integer(cf2$pi0>0.5) != d$y)
          }
        }
        
        if (!all(is.na(best$GCV$beta))) {
          B_rte[r,]  <- best$GCV$beta
          Fsup_rte[r] <- max(abs(best$GCV$f - d$ft))
          MCR_rte[r]  <- mean(as.integer(best$GCV$pi>0.5) != d$y)
        }
        

        # Parametric ridge — data-driven k via GCV
        pr_best_gcv <- Inf
        for (k in k_grid) {
          pr <- tryCatch(fit_par_ridge(d$y, d$X, k), error = function(e) NULL)
          if (is.null(pr)) next
          om_pr <- pr$pi_hat * (1 - pr$pi_hat)
          XtOX  <- t(d$X) %*% (om_pr * d$X)
          H_pr  <- d$X %*% solve(XtOX + k * diag(p), t(d$X) * om_pr)
          resid <- d$y - pr$pi_hat
          RSS   <- sum(resid^2)
          tr_H  <- sum(diag(H_pr))
          gcv_pr <- (RSS / n) / max((1 - tr_H / n)^2, 1e-12)
          if (is.finite(gcv_pr) && gcv_pr < pr_best_gcv) {
            pr_best_gcv <- gcv_pr
            B_pr[r, ] <- pr$beta
            MCR_pr[r] <- mean(as.integer(pr$pi_hat > 0.5) != d$y)
          }
        }
        
        CR_smse[r,"GCV"]  <- if(!all(is.na(best$GCV$beta)))  sum((best$GCV$beta-bt)^2)  else NA
        CR_smse[r,"AICc"] <- if(!all(is.na(best$AICc$beta))) sum((best$AICc$beta-bt)^2) else NA
        CR_smse[r,"BIC"]  <- if(!all(is.na(best$BIC$beta)))  sum((best$BIC$beta-bt)^2)  else NA
      }
      
      # Aggregate
      ok_r <- complete.cases(B_rte)
      ok_l <- complete.cases(B_lse)
      ok_p <- complete.cases(B_pr)
      sq <- function(B,ok) if(sum(ok)>1) median(rowSums(sweep(B[ok,,drop=F],2,bt)^2)) else NA
      b2 <- function(B,ok) if(sum(ok)>1) sum((colMeans(B[ok,,drop=F])-bt)^2) else NA
      
      smse_r <- sq(B_rte,ok_r); smse_l <- sq(B_lse,ok_l); smse_p <- sq(B_pr,ok_p)
      b2_r <- b2(B_rte,ok_r);   b2_l <- b2(B_lse,ok_l)
      
      kp_m <- colMeans(KP_smse, na.rm=TRUE)
      kp_b2 <- kp_v <- rep(NA, length(k_grid))
      for (ik in seq_along(k_grid)) {
        bk <- KP_beta[,ik,]; ok_k <- complete.cases(bk)
        if (sum(ok_k)>1) {
          kp_b2[ik] <- sum((colMeans(bk[ok_k,,drop=F])-bt)^2)
          kp_v[ik]  <- kp_m[ik] - kp_b2[ik]
        }
      }
      
      cv <- rep(NA, p)
      if (sum(ok_r) > 5) {
        for (j in 1:p) {
          se <- sd(B_rte[ok_r,j])
          lo <- B_rte[ok_r,j] - 1.96*se
          hi <- B_rte[ok_r,j] + 1.96*se
          cv[j] <- mean(lo<=bt[j] & bt[j]<=hi)
        }
      }
      
      cat(sprintf("SMSE: R=%.4f L=%.4f P=%.4f\n", smse_r, smse_l, smse_p))
      
      all_res[[tag]] <- list(
        p=p, n=n, rho=rho, tag=tag,
        smse=c(IWSRTE=smse_r, IWSLSE=smse_l, ParRidge=smse_p),
        bias2=c(IWSRTE=b2_r, IWSLSE=b2_l),
        var=c(IWSRTE=smse_r-b2_r, IWSLSE=smse_l-b2_l),
        kp_k=k_grid, kp_smse=kp_m, kp_bias2=kp_b2, kp_var=kp_v,
        cov=cv,
        fsup=c(IWSRTE=median(Fsup_rte,na.rm=T), IWSLSE=median(Fsup_lse,na.rm=T)),
        mcr=c(IWSRTE=mean(MCR_rte,na.rm=T), IWSLSE=mean(MCR_lse,na.rm=T),
              ParRidge=mean(MCR_pr,na.rm=T)),
        crit_smse=colMeans(CR_smse,na.rm=T),
        B_rte=B_rte, ok_rte=ok_r)
    }
  }
}

# ── 9. TABLES ─────────────────────────────────────────────────

cat("\n=== Tables ===\n")
t1 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             IWSRTE=round(r$smse["IWSRTE"],6),
             IWSLSE=round(r$smse["IWSLSE"],6),
             ParRidge=round(r$smse["ParRidge"],6))))
rownames(t1)<-NULL; write.csv(t1,"tables/table1_smse.csv",row.names=F)

t2 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             Bias2_RTE=round(r$bias2["IWSRTE"],6),Var_RTE=round(r$var["IWSRTE"],6),
             SMSE_RTE=round(r$smse["IWSRTE"],6),
             Bias2_LSE=round(r$bias2["IWSLSE"],6),Var_LSE=round(r$var["IWSLSE"],6),
             SMSE_LSE=round(r$smse["IWSLSE"],6))))
rownames(t2)<-NULL; write.csv(t2,"tables/table2_biasvar.csv",row.names=F)

t3 <- do.call(rbind, lapply(all_res, function(r) {
  df <- data.frame(p=r$p, n=r$n, rho=r$rho)
  for(j in seq_along(r$cov)) df[[paste0("beta",j)]] <- round(r$cov[j],3)
  if(length(r$cov) < max(p_vec)) {
    for(j in (length(r$cov)+1):max(p_vec))
      df[[paste0("beta",j)]] <- NA
  }
  df
}))
rownames(t3)<-NULL; write.csv(t3,"tables/table3_coverage.csv",row.names=F)

t4 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             IWSRTE=round(r$mcr["IWSRTE"],4),
             IWSLSE=round(r$mcr["IWSLSE"],4),
             ParRidge=round(r$mcr["ParRidge"],4))))
rownames(t4)<-NULL; write.csv(t4,"tables/table4_mcr.csv",row.names=F)

t5 <- do.call(rbind, lapply(all_res, function(r)
  data.frame(p=r$p,n=r$n,rho=r$rho,
             GCV=round(r$crit_smse["GCV"],6),
             AICc=round(r$crit_smse["AICc"],6),
             BIC=round(r$crit_smse["BIC"],6))))
rownames(t5)<-NULL; write.csv(t5,"tables/table5_criteria.csv",row.names=F)
cat("Tables saved.\n")

# ── 10. FIGURES ──────────────────────────────────────────────

cat("=== Figures ===\n")
C1<-"#2166AC"; C2<-"#B2182B"; C3<-"#4DAF4A"

# k-path plots
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

# QQ plots
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

# f-hat convergence
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

# SMSE dot plot — log scale
for(p in p_vec) {
  sub <- t1[t1$p==p,]; if(nrow(sub)==0) next
  png(sprintf("figures/smse_dots_p%d.png",p), width=900, height=500, res=120)
  par(mar=c(7,5,3,2))
  
  nc <- nrow(sub)
  labs <- sprintf("n=%d, rho=%.3f", sub$n, sub$rho)
  
  yr <- range(c(sub$IWSRTE, sub$IWSLSE, sub$ParRidge), na.rm=TRUE)
  yr[1] <- max(yr[1], 1e-4)  # floor
  
  plot(1:nc, sub$IWSRTE, log="y", pch=19, cex=1.4, col=C1,
       xlim=c(0.5, nc+0.5), ylim=yr,
       xlab="", ylab="SMSE (log scale)", xaxt="n",
       main=bquote("SMSE Comparison: p ="~.(p)))
  points(1:nc, sub$IWSLSE,   pch=17, cex=1.4, col=C2)
  points(1:nc, sub$ParRidge, pch=15, cex=1.4, col=C3)
  axis(1, at=1:nc, labels=labs, las=2, cex.axis=0.7)
  abline(v=1:nc, col="gray90", lty=3)
  legend("topright",
         legend=c("IWSRTE","IWSLSE","ParRidge"),
         pch=c(19,17,15), col=c(C1,C2,C3), bty="n", cex=0.9)
  dev.off()
}

# MCR bars
for(p in p_vec) {
  sub<-t4[t4$p==p,]; if(nrow(sub)==0) next
  png(sprintf("figures/mcr_bars_p%d.png",p),width=900,height=500,res=120)
  par(mar=c(6,4.5,3,8),xpd=TRUE)
  vals<-t(as.matrix(sub[,c("IWSRTE","IWSLSE","ParRidge")]))
  labs<-sprintf("n=%d\nrho=%.3f",sub$n,sub$rho)
  barplot(vals,beside=T,col=c(C1,C2,C3),names.arg=labs,las=2,cex.names=0.65,
          ylab="MCR",main=bquote("Misclassification Rate: p ="~.(p)))
  legend("topright",inset=c(-0.18,0),legend=c("IWSRTE","IWSLSE","ParRidge"),
         fill=c(C1,C2,C3),bty="n",cex=0.8)
  dev.off()
}


# Fitted f_hat vs true f(t) — one per config, GCV-optimal k
for(tag in names(all_res)) {
  r <- all_res[[tag]]
  bt <- beta_list[[as.character(r$p)]]
  
  set.seed(2025 + which(names(all_res) == tag))
  d <- gen_data(r$n, r$p, r$rho, bt)
  
  h_mid <- median(h_grid)
  S_h <- build_smoother(d$t, h_mid)
  
  # Find GCV-optimal k
  best_gcv_val <- Inf
  best_k <- 0
  for (k_try in k_grid) {
    rk_try <- tryCatch(ridge_at_k(d$y, d$X, S_h, k_try), error=function(e) NULL)
    if (is.null(rk_try) || !rk_try$ok) next
    cr_try <- tryCatch(sel_crit(d$y, rk_try$P_hk),
                       error=function(e) c(GCV=Inf))
    if (is.finite(cr_try["GCV"]) && cr_try["GCV"] < best_gcv_val) {
      best_gcv_val <- cr_try["GCV"]
      best_k <- k_try
    }
  }
  
  # Fit at k=0 (IWSLSE) and k=best_k (IWSRTE)
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
         legend=c(expression("True " * f(t)),
                  "IWSLSE (k = 0)",
                  bquote("IWSRTE (k = "*.(round(best_k,2))*")")),
         col=c("black", C2, C1), lty=c(1, 2, 1), lwd=c(2.5, 2, 2),
         bty="n", cex=0.8)
  dev.off()
}



# Decision boundary plot in (x1, t) plane — GCV-optimal k
for(tag in names(all_res)) {
  r <- all_res[[tag]]
  bt <- beta_list[[as.character(r$p)]]
  
  set.seed(2025 + which(names(all_res) == tag))
  d <- gen_data(r$n, r$p, r$rho, bt)
  
  h_mid <- median(h_grid)
  S_h <- build_smoother(d$t, h_mid)
  
  # GCV-optimal k for IWSRTE
  best_gcv_val <- Inf
  best_k <- 0
  for (k_try in k_grid) {
    rk_try <- tryCatch(ridge_at_k(d$y, d$X, S_h, k_try), error=function(e) NULL)
    if (is.null(rk_try) || !rk_try$ok) next
    cr_try <- tryCatch(sel_crit(d$y, rk_try$P_hk),
                       error=function(e) c(GCV=Inf))
    if (is.finite(cr_try["GCV"]) && cr_try["GCV"] < best_gcv_val) {
      best_gcv_val <- cr_try["GCV"]
      best_k <- k_try
    }
  }
  
  fit_rte <- tryCatch(ridge_at_k(d$y, d$X, S_h, best_k), error=function(e) NULL)
  fit_lse <- tryCatch(ridge_at_k(d$y, d$X, S_h, 0),      error=function(e) NULL)
  
  # GCV-optimal k for ParRidge
  pr_best_gcv <- Inf; pr_best_k <- 0
  for (k_try in k_grid) {
    pr_try <- tryCatch(fit_par_ridge(d$y, d$X, k_try), error=function(e) NULL)
    if (is.null(pr_try)) next
    om_pr <- pr_try$pi_hat * (1 - pr_try$pi_hat)
    H_pr <- d$X %*% solve(t(d$X)%*%(om_pr*d$X) + k_try*diag(r$p), t(d$X)*om_pr)
    rss <- sum((d$y - pr_try$pi_hat)^2)
    gcv_pr <- (rss/r$n) / max((1 - sum(diag(H_pr))/r$n)^2, 1e-12)
    if (is.finite(gcv_pr) && gcv_pr < pr_best_gcv) {
      pr_best_gcv <- gcv_pr; pr_best_k <- k_try
    }
  }
  fit_pr <- tryCatch(fit_par_ridge(d$y, d$X, pr_best_k), error=function(e) NULL)
  
  if (is.null(fit_rte) || is.null(fit_lse) || is.null(fit_pr)) next
  
  # Grid in (t, x1) space, other x's at column means
  ng <- 150
  t_grid  <- seq(0, 1, length.out=ng)
  x1_range <- range(d$X[,1]) * 1.2
  x1_grid <- seq(x1_range[1], x1_range[2], length.out=ng)
  x_bar <- colMeans(d$X)
  
  # True boundary: x1 = -(x_bar[-1]'beta[-1] + f(t)) / beta[1]
  offset_true <- sum(x_bar[-1] * bt[-1])
  db_true <- -(offset_true + f_true(t_grid)) / bt[1]
  
  # IWSRTE boundary
  f_interp_rte <- approx(d$t[order(d$t)], fit_rte$f_hat[order(d$t)],
                         xout=t_grid, rule=2)$y
  offset_rte <- sum(x_bar[-1] * fit_rte$beta[-1])
  db_rte <- -(offset_rte + f_interp_rte) / fit_rte$beta[1]
  
  # IWSLSE boundary
  f_interp_lse <- approx(d$t[order(d$t)], fit_lse$f_hat[order(d$t)],
                         xout=t_grid, rule=2)$y
  offset_lse <- sum(x_bar[-1] * fit_lse$beta[-1])
  db_lse <- -(offset_lse + f_interp_lse) / fit_lse$beta[1]
  
  # ParRidge boundary (no f, flat line)
  offset_pr <- sum(x_bar[-1] * fit_pr$beta[-1])
  db_pr <- rep(-offset_pr / fit_pr$beta[1], ng)
  
  # Predicted probability grid (IWSRTE)
  prob_grid <- matrix(NA, ng, ng)
  for (i in seq_along(t_grid)) {
    for (j in seq_along(x1_grid)) {
      x_new <- x_bar
      x_new[1] <- x1_grid[j]
      eta_new <- sum(x_new * fit_rte$beta) + f_interp_rte[i]
      prob_grid[i, j] <- expit(eta_new)
    }
  }
  
  png(sprintf("figures/decision_%s.png", tag), width=800, height=600, res=120)
  par(mar=c(4.5, 4.5, 3, 5))
  
  # Background heatmap
  image(t_grid, x1_grid, prob_grid,
        col=colorRampPalette(c("#2166AC","#F7F7F7","#B2182B"))(100),
        xlab="t (nonparametric covariate)",
        ylab=expression(x[1] ~ " (parametric covariate)"),
        main=bquote("Decision boundaries: " ~
                      p==.(r$p) ~ ", " ~ n==.(r$n) ~ ", " ~ rho==.(r$rho)),
        zlim=c(0, 1))
  
  # 0.5 contour
  contour(t_grid, x1_grid, prob_grid, levels=0.5,
          add=TRUE, drawlabels=FALSE, col=C1, lwd=1, lty=3)
  
  # Boundaries
  in_true <- db_true >= x1_range[1] & db_true <= x1_range[2]
  lines(t_grid[in_true], db_true[in_true], lwd=3, col="black")
  
  in_rte <- db_rte >= x1_range[1] & db_rte <= x1_range[2]
  lines(t_grid[in_rte], db_rte[in_rte], lwd=2.5, col=C1)
  
  in_lse <- db_lse >= x1_range[1] & db_lse <= x1_range[2]
  lines(t_grid[in_lse], db_lse[in_lse], lwd=2, col=C2, lty=2)
  
  lines(t_grid, db_pr, lwd=2, col=C3, lty=4)
  
  # Data points
  points(d$t[d$y==1], d$X[d$y==1, 1], pch=16, cex=0.7,
         col=adjustcolor("#B2182B", 0.6))
  points(d$t[d$y==0], d$X[d$y==0, 1], pch=4, cex=0.7,
         col=adjustcolor("#2166AC", 0.6))
  
  legend("bottomright", bg="white", cex=0.7,
         legend=c("True boundary",
                  bquote("IWSRTE (k="*.(round(best_k,2))*")"),
                  "IWSLSE (k=0)",
                  bquote("ParRidge (k="*.(round(pr_best_k,2))*")"),
                  "y = 1", "y = 0"),
         col=c("black", C1, C2, C3,
               adjustcolor("#B2182B",0.6), adjustcolor("#2166AC",0.6)),
         lty=c(1, 1, 2, 4, NA, NA),
         pch=c(NA, NA, NA, NA, 16, 4),
         lwd=c(3, 2.5, 2, 2, NA, NA))
  dev.off()
}
