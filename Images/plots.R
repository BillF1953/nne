slr =read.csv("file:///C:/Users/Tyler/coveragestudymentchandhooker/estimates.csv", header=TRUE)
nslr2 = read.csv("file:///C:/Users/Tyler/coveragestudymentchandhooker/estimates_nonlinear_153_3000.csv", header=TRUE)
slr = slr[order(slr$Estimate), ]
nslr2$TCI = nslr2$Estimate  +  2*nslr2$SE
nslr2$LCI = nslr2$Estimate  -  2*nslr2$SE
hist(slr$Estimate, xlab="Point Estimate Theta=20",main="SLR")
hist(4*slr$SE, xlab="2-Sigma CI Width", main="Coverage Study NSLR2")
hist(6*slr$SE, xlab="3-Sigma CI Width", main="Coverage Study NSLR2")

mean(slr$Estimate)
mean(slr$SE)

hist(slr$TCI)
hist(slr$LCI)



hist(nslr2$Estimate, xlab="Point Estimate Theta=33.4",main="SNLR2")
hist(4*nslr2$SE, xlab="2-Sigma CI Width", main="Coverage Study NSLR2")
hist(6*nslr2$SE, xlab="3-Sigma CI Width", main="Coverage Study NSLR2")
