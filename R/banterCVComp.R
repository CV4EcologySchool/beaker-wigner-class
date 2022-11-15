# banter for check
library(PAMpal)
library(banter)

acst <- readRDS('rds/PASCAL_Base_AcSt.rds')
acst <- filter(acst, species %in% c('ZC', 'BW43', 'MS', 'BB', 'BW37V'))
acst <- calculateICI(acst, time='peakTime')
# cces <- readRDS('rds/ccesStudy.rds')
# cces <- filter(cces, species %in% c('ZC', 'BW43', 'MS', 'BB', 'BW37V'))
# cces576 <- readRDS('rds/ccesStudy576.rds')
# cces <- bindStudies(cces, cces576)
# cces <- calculateICI(cces, time='peakTime')
cces <- readRDS('rds/ccesStudyFixed.rds')
combined <- bindStudies(acst, cces)

csv <- read.csv('Combined_All_10k.csv', stringsAsFactors = F)
csv <- dplyr::filter(csv, species %in% c('ZC', 'BW43', 'MS', 'BB', 'BW37V'))

bntData <- export_banter(combined[unique(csv$eventId[csv$trainCat %in% c(1)])],
                         dropVars = c('Latitude', 'Longitude'))

bntMdl <- initBanterModel(bntData$events)
bntMdl <- addBanterDetector(bntMdl, bntData$detectors, ntree=5e3, sampsize=7, importance=TRUE, num.cores=1)
bntMdl <- runBanterModel(bntMdl, ntree=20e3, sampsize=4)
summary(bntMdl)

unique(csv$eventId[csv$trainCat == 2])[!unique(csv$eventId[csv$trainCat == 2]) %in% names(events(combined))]
predData <- export_banter(combined[unique(csv$eventId[csv$trainCat == 2])], dropVars=c('Latitude', 'Longitude'))

pred <- predict(bntMdl, predData)
pred
pdf <- pred$predict.df
pdf$p <- 0
for(i in 1:nrow(pdf)) {
    pdf$p[i] <- pdf[i, pdf$predicted[i]]
}
ggplot(pdf) +
    geom_histogram(aes(x=p, fill=correct), bins=50)
rfPermute::plotImportance(bntMdl@model)
plotDetectorTrace(bntMdl)

# trained on train+val
bnt12 <- bntMdl
bnt1 <- bntMdl
bn13 <- bntMdl
bnt123 <- bntMdl
