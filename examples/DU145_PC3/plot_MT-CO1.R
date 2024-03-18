# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
input_path_1 = args[[1]]
input_path_2 = args[[2]]
input_path_3 = args[[3]]
output_path = args[[4]]

library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

type_colors = c("#4a8337", "#6bac21", "#ddd48f", "#cda989",
   "#b07000", "#804000")

expr = as.matrix(read.table(input_path_1, header=TRUE))
cellavi_probs = as.matrix(read.table(input_path_2))
prodLDA_probs = as.matrix(read.table(input_path_3))


MTCO1.orig = expr[,10844]

# Sample values for prodLDA (PyTorch does not allow sampling
# of multinomial with different sizes).
prodLDA_samples = matrix(rbinom(
      n=length(prodLDA_probs),
      prob=t(prodLDA_probs),
      size=as.integer(rowSums(expr))),
   byrow=TRUE,
   nrow=nrow(prodLDA_probs)
)

# Key to the types.
# 1: DU145/Res
# 2: DU145/Sen
# 3: PC3/Res
# 4: PC3/Sen
# 5: LNCaP - androgen
# 5: LNCaP + androgen

type = c(
   2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
   2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,
   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,
   4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
   4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
   4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
   3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
   3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,
   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
   5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
   6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6
)

x = rep(seq(1, ncol(Cellavi_samples)), nrow(Cellavi_samples))
Cellavi_y = as.vector(t(Cellavi_samples)) / 1000
prodLDA_y = t(prodLDA_samples) / 1000

pdf(output_path, height = 8, width = 8, useDingbats=FALSE)
showtext_begin()

par(mfrow=c(2,1))
par(mar=c(2.8,2.8,0.7,0))

plot(x, Cellavi_y, ylim=c(0, 100), type="n", bty="n", ylab="", xlab="",
     xaxt="n", yaxt="n", panel.first=grid())

points(x, Cellavi_y, pch=".", col=type_colors[type])
points(MTCO1.orig / 1000, pch=19, cex=.3, col="black")

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
     at=c(0, 72, 144, 234, 323, 419, 467))
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Cell index", line=1.5, col.lab="gray30", family="Avenir Medium")
title(ylab="MT-CO1 reads (x1000)", line=1.5, col.lab="gray30",
     family="Avenir Medium")

plot(x, prodLDA_y, ylim=c(0, 100), type="n", bty="n", ylab="", xlab="",
     xaxt="n", yaxt="n", panel.first=grid())

points(x, prodLDA_y, pch=".", col=type_colors[type])
points(MTCO1.orig / 1000, pch=19, cex=.3, col="black")

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
     at=c(0, 72, 144, 234, 323, 419, 467))
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Cell index", line=1.5, col.lab="gray30", family="Avenir Medium")
title(ylab="MT-CO1 reads (x1000)", line=1.5, col.lab="gray30",
     family="Avenir Medium")

legend(x="topright", inset=.01,
     bg="white", box.col="gray50",
     col=type_colors[unique(type)], pch=19, cex=.8,
     legend=c("DU145/Res", "DU145/Sen", "PC3/Res", "PC3/Sen",
         "LNCaP/+andro", "LNCaP/-andro")
)

showtext_end()
dev.off()
