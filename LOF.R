#####?e?L?X?g?t?@?C???????f?[?^???ǂ???#####
Dataset <- read.table("/pathway/to/the/directory/LOF_ROC.txt", header=TRUE, 
  sep="\t", na.strings=c("", "NA"), dec=".", fill=TRUE, 
  quote="\"", comment.char="", strip.white=TRUE)
Dataset <- read.table("/pathway/to/the/directory/LOF_ROC.txt", header=TRUE, sep="\t", 
  na.strings=c("", "NA"), dec=".", fill=TRUE, quote="\"", comment.char="", strip.white=TRUE)
library(pROC, pos=17)
#####???ʌ????̐f?f?ւ̐??m?x?̕]??(ROC?Ȑ??j#####
ROC <- NULL
ROC <- roc(Known~LOF, data=Dataset, ci=TRUE, direction="auto")
if(ROC$thresholds[1]==-Inf){thre <- c(unique(sort(ROC$predictor)), Inf)}
if(ROC$thresholds[1]==Inf){thre <- c(unique(sort(ROC$predictor, decreasing=TRUE)), -Inf)}
windows(width=7, height=7); par(lwd=1, las=1, family="sans", cex=1, mgp=c(3.0,1,0))
plot(thre, ROC$sensitivities, ylim=c(0,1), type="l", ylab="Sensitivity/Specificity", xlab="Threshold")
par(new=T)
plot(thre, ROC$specificities, ylim=c(0,1), type="l", lty=2, ylab="", xlab="", col.axis=0)
legend("bottom", horiz=TRUE, c("Sensitivity", "Specificity"), lty=1:2, box.lty=0)
windows(width=7, height=7); par(lwd=1, las=1, family="sans", cex=1, mgp=c(3.0,1,0))
co <- pROC::coords(ROC, "best", best.method="youden", best.weights=c(1, 0.5), transpose = FALSE)
if(ROC$thresholds[1]==-Inf){co[,1] <- min(ROC$predictor[ROC$predictor>co[,1]])}
if(ROC$thresholds[1]==Inf)co[,1] <- max(ROC$predictor[ROC$predictor<co[,1]])
plot(ROC, print.thres=co[,1], grid=TRUE)
if(ROC$thresholds[1]==-Inf){pROC::coords(ROC, x=c(-Inf, unique(sort(ROC$predictor)), Inf), transpose = FALSE)}
if(ROC$thresholds[1]==Inf){pROC::coords(ROC, x=c(Inf, unique(sort(ROC$predictor, decreasing=TRUE)), -Inf), transpose = FALSE)}
### ???\???l??臒l?u?ȏ??v???z???Ɣ??肵???ꍇ?̊??x?A?��ٓx?����??B
ROC
cat(gettextRcmdr("Area under the curve"), signif(ROC$auc[1], digits=3), gettextRcmdr("95% CI"), signif(ROC$ci[1], digits=3), "-", signif(ROC$ci[3], digits=3), "
")