# Purpose: Perform lasso regression (iterated, single-response) using emg features as observed
# variables and force features as response variables.

library(RColorBrewer)
library(tibble)
library(gridExtra)
library(corrgram)
library(glmnet)
library(plyr) # watch load order with dplyr
library(dplyr)
 
# filename_addf has f3f2,f3f1 calcs. 
# take care if incorporating -- reorder cols changes.

setwd('~/dropbox/nasa_stretch/force_features')
data = read.csv('~/dropbox/nasa_stretch/force_features/force_emg_expl.csv')

# This script outputs graphics to crossval and lassoplots
dir.create("~/dropbox/nasa_stretch/force_features/lassoplots")
dir.create("~/dropbox/nasa_stretch/force_features/crossval")

# and elastic net graphics to enetcrossvals and enetplots
dir.create("~/dropbox/nasa_stretch/force_features/enetplots")
dir.create("~/dropbox/nasa_stretch/force_features/enetcrossvals")

# calculate the bilateral ratios (4 total)
data$bmg_wav =(data$lmg_airsum + data$rmg_airsum)/(data$lmg_lsrsum + data$rmg_lsrsum)
data$bmg_iemg =(data$lmg_iemg_air + data$rmg_iemg_air)/(data$lmg_iemg_lnd + data$rmg_iemg_lnd)
data$bta_wav =(data$lta_airsum +data$rta_airsum)/(data$lta_lsrsum+ data$rta_lsrsum)
data$bta_iemg = (data$lta_iemg_air + data$rta_iemg_air)/(data$lta_iemg_lnd + data$rta_iemg_lnd)

data = data[,c(1:4,5:20, 31:34, 21:30)] #re-order cols. lhs = emg, rhs = force vars


# split data frames by platform

sep_plat = dlply(data, "Platform", identity)

df.control = sep_plat[[1]]
df.flywheel = sep_plat[[2]]
df.iss = sep_plat[[3]]
df.shuttle = sep_plat[[4]]
df.treatmenta = sep_plat[[5]]
df.treatmentb = sep_plat[[6]]

# Standardize iss predictor variables
scale(df.iss[,5:24])


# Split ISS cases
day_split_iss = dlply(df.iss,"normTime", identity)
# 1-A; 2-B; 3-C; 4-E; 5-F; 6-G

##  output all images to files

# Prettified grid to table
png(filename = "data_output.png", height=11, width=14, units = "in", res=150)
grid.table(df.iss[1:30,c(1:10,26:28)])
dev.off()

# study correlograms for various cases, see if they very based on JDT days.
png(filename = 'corrs/corrgram.png', width=13.3, height=7.5, units = "in", res = 300)
corrgram(df.iss[,5:24], upper.panel = panel.pts)
title(main = "Design Matrix Correlogram ISS")
dev.off()

for(k in 1:6){
  png(filename = paste('corrs/correlogram_ISS_', names(day_split_iss)[k], ".png", sep="", collapse=""),
      width = 13.3, height=7.5, units ="in", res = 150)
  corrgram(day_split_iss[[k]][,5:24], upper.panel = panel.pts)
  title(main = paste("Design Matrix Correlogram ISS", names(day_split_iss)[k], collapse=""))
  dev.off()
}

palette(brewer.pal(12,"Paired"))

# suppress NA to 0 for F3F2 which has large number of such values
data[is.na(data)] = 0

# lasso loops for entire ISS data (days unseparated)
for(i in 25:34){
  print(i)
  fit = glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss[,i]),
               lambda = cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss[,i]))$lambda.3se)
  
  png(filename = paste("lassoplots/ResponseVariable_", colnames(df.iss)[i] ,".png", sep="", collapse=""),
      res =150, width=6.5, height=6.5, units="in")             
  
  plot(fit, lwd = 2, label = FALSE)
  par(mar=c(4.5,4.5,1,4))
  vn=colnames(data)[5:24]
  vnat=coef(fit)
  vnat=vnat[-1,ncol(vnat)] # remove the intercept, and get the coefficients at the end of the path
  axis(4, at=vnat,line=-3,label=vn,las=1,tick=FALSE, cex.axis=1) 
  title(main = paste("Response Variable ", colnames(df.iss)[i]))
  dev.off()
  
  png(filename = paste("crossvals/CV_", colnames(df.iss)[i],".png", sep = "", collapse=""),
      res=150, width=6.5, height=6.5, units="in")
  cvfit = cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss[,i]))
  plot(cvfit)
  title(main = paste("Cross-Validation Response Variable ", colnames(df.iss)[i]),line=2)
  dev.off()
}

# lasso loops for normTime
for(k in 1:6){
  for(i in 25:34){
    fit = glmnet(data.matrix(day_split_iss[[k]][,5:24]), data.matrix(day_split_iss[[k]][,i]), 
                 lambda = cv.glmnet(data.matrix(day_split_iss[[k]][,5:24]), data.matrix(day_split_iss[[k]][,i]))$lambda.3se)
    
    png(filename = paste("lassoplots/ResponseVariable_", colnames(day_split_iss[[k]])[i],"_ISS ",names(day_split_iss)[k],".png", sep="", collapse=""),
        res =150, width=6.5, height=6.5, units="in")             
    
    plot(fit, lwd=2, label = FALSE)
    par(mar=c(4.5,4.5,1,4))
    vn=colnames(data)[5:24]
    vnat=coef(fit)
    vnat=vnat[-1,ncol(vnat)] # remove the intercept, and get the coefficients at the end of the path
    axis(4, at=vnat,line=-3,label=vn,las=1,tick=FALSE, cex.axis=1) 
    title(main = paste("Response Variable ", colnames(day_split_iss[[k]])[i],"_ISS ", names(day_split_iss[k])))
    dev.off()
    
    png(filename = paste("crossvals/CV_", colnames(day_split_iss[[k]])[i],"ISS ",names(day_split_iss[k]),".png", sep = "", collapse=""),
        res=150, width=6.5, height=6.5, units="in")
    cvfit = cv.glmnet(data.matrix(day_split_iss[[k]][,5:24]), data.matrix(day_split_iss[[k]][,i]))
    plot(cvfit)
    title(main = paste("Cross-Validation Response Variable ", colnames(day_split_iss[[k]])[i], "_ISS ", names(day_split_iss[k])), line=2)
    dev.off()
    
  }
}

# elastic net, start with alpha = 0.5. Same loops. Exploratory. Days Separated.

for(k in 1:6){
  for(i in 25:34){
    fit = glmnet(data.matrix(day_split_iss[[k]][,5:24]), data.matrix(day_split_iss[[k]][,i]), 
                 lambda = cv.glmnet(data.matrix(day_split_iss[[k]][,5:24]), data.matrix(day_split_iss[[k]][,i]))$lambda.3se,
                 alpha=0.5)
    
    png(filename = paste("enetplots/ResponseVariable_", colnames(day_split_iss[[k]])[i],"_ISS ",names(day_split_iss)[k],".png", sep="", collapse=""),
        res =150, width=6.5, height=6.5, units="in")             
    
    plot(fit, lwd=2, label = FALSE)
    par(mar=c(4.5,4.5,1,4))
    vn=colnames(data)[5:24]
    vnat=coef(fit)
    vnat=vnat[-1,ncol(vnat)] # remove the intercept, and get the coefficients at the end of the path
    axis(4, at=vnat,line=-3,label=vn,las=1,tick=FALSE, cex.axis=1) 
    title(main = paste("Response Variable ", colnames(day_split_iss[[k]])[i],"_ISS ", names(day_split_iss[k])))
    dev.off()
    
    png(filename = paste("enetcrossvals/CV_", colnames(day_split_iss[[k]])[i],"ISS ",names(day_split_iss[k]),".png", sep = "", collapse=""),
        res=150, width=6.5, height=6.5, units="in")
    cvfit = cv.glmnet(data.matrix(day_split_iss[[k]][,5:24]), data.matrix(day_split_iss[[k]][,i]))
    plot(cvfit)
    title(main = paste("Cross-Validation Response Variable ", colnames(day_split_iss[[k]])[i], "_ISS ", names(day_split_iss[k])), line=2)
    dev.off()
    
  }
}

# elastic net alpha = 0.5 exploratory. unseparated days.

for(i in 25:34) {
  fit = glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss[,i]), 
               lambda = cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss[,i]))$lambda.3se,
               alpha = 0.5)
  
  png(filename = paste("enetplots/ResponseVariable_", colnames(df.iss)[i] ,".png", sep="", collapse=""),
      res =150, width=6.5, height=6.5, units="in")             
  
  plot(fit, lwd = 2, label = FALSE)
  par(mar=c(4.5,4.5,1,4))
  vn=colnames(data)[5:24]
  vnat=coef(fit)
  vnat=vnat[-1,ncol(vnat)] # remove the intercept, and get the coefficients at the end of the path
  axis(4, at=vnat,line=-3,label=vn,las=1,tick=FALSE, cex.axis=1) 
  title(main = paste("Response Variable ", colnames(df.iss)[i]))
  dev.off()
  
  png(filename = paste("enetcrossvals/CV_", colnames(df.iss)[i],".png", sep = "", collapse=""),
      res=150, width=6.5, height=6.5, units="in")
  cvfit = cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss[,i]))
  plot(cvfit)
  title(main = paste("Cross-Validation Response Variable ", colnames(df.iss)[i]),line=2)
  dev.off()
}


# Show example of F2F1 LASSO and ELNET performance.

foldid = sample(1:10,size=length(df.iss$F2F1),replace=TRUE)
cv1=cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss$F2F1),
              foldid=foldid,alpha=1)

cv.8=cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss$F2F1),
              foldid=foldid,alpha=.8)

cv.5=cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss$F2F1),
              foldid=foldid,alpha=.5)

cv.2=cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss$F2F1),
              foldid=foldid,alpha=.2)

cv0=cv.glmnet(data.matrix(df.iss[,5:24]), data.matrix(df.iss$F2F1),
              foldid=foldid,alpha=0)

png(filename = paste("enetcrossvals/CV_","alpha_behavior.png", sep = "", collapse=""),
    res=300, width=6.5, height=6.5, units="in")
plot(log(cv1$lambda),cv1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv1$name, cex.lab = 1.3)
points(log(cv.8$lambda),cv.8$cvm,pch=19,col="black")
points(log(cv.5$lambda),cv.5$cvm,pch=19,col="grey")
points(log(cv.2$lambda),cv.2$cvm,pch=19,col="violet")
points(log(cv0$lambda),cv0$cvm,pch=19,col="blue")
legend("topright",legend=c("alpha = 1","alpha = 0.8","alpha = 0.5","alpha = 0.2","alpha = 0"),pch=19,
       col=c("red","black","grey","violet","blue"))
title(main = "Regularization Behavior for Range of Alpha", cex.main = 1.5)
dev.off()

