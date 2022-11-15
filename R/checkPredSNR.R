# PREDS INVESTIGATION BY SNR
cvPreds <- read.csv('../../CV4Ecology/preds/pred_comb_norm/Combined_BW_Val_CombOrigpred.csv', stringsAsFactors = F)
library(dplyr)
library(ggplot2)
library(patchwork)

cvPreds %>%
    ggplot() +
    geom_density(aes(x=snr,, col=pred == true)) +
    facet_wrap(~true) +
    xlim(-5, 50)

cvPreds %>%
    filter(pred == 0, true != 0) %>%
    ggplot(aes(col=as.character(true))) +
    # geom_density(aes(x=p3)) +
    # geom_density(aes(x=p0))
    geom_point(aes(x=snr, y=p0))

doSnrHist <- function(x, by=c('true', 'pred'), cut=2, crange=c(-5, 50)) {
    outList <- vector('list', length=length(unique(x$true)) + 1)
    brks <- seq(from=crange[1], to=crange[2], by=cut)
    x <- mutate(x, cuts = cut(snr, breaks = brks, labels=brks[-1]))
    # doBy <- enquo(by)
    outList[[1]] <- x %>%
        group_by(cuts) %>%
        summarise(probPos = sum(true == pred)/n(),
                  nPos = sum(true == pred),
                  nTot = n()) %>%
        ggplot() +
        geom_bar(aes(x=cuts, y=nTot), stat='identity', fill='red') +
        geom_bar(aes(x=cuts, y=nPos), col='green', stat='identity') +
        geom_text(aes(x=cuts, y=nTot + max(nTot) * .05, label= paste0(100-round(100 * probPos, 0), '%'))) +
        ggtitle(paste0('All Error % by SNR'))

    for(i in unique(x$true)) {
        outList[[i+2]] <- x %>%
            filter(!!as.symbol(by) == i) %>%
            group_by(cuts) %>%
            summarise(probPos = sum(true == pred)/n(),
                      nPos = sum(true == pred),
                      nTot = n()) %>%
            ggplot() +
            geom_bar(aes(x=cuts, y=nTot), stat='identity', fill='red') +
            geom_bar(aes(x=cuts, y=nPos), col='green', stat='identity') +
            geom_text(aes(x=cuts, y=nTot + max(nTot) * .05, label= paste0(100-round(100 * probPos, 0), '%'))) +
            ggtitle(paste0('"', by, '" Species ', i)) +
            scale_fill_discrete(drop=FALSE) +
            scale_x_discrete(drop=FALSE)
    }
    outList
}
predDir <- '../../CV4Ecology/preds/pred_comb_norm_snr5/'
valPred <- list.files(predDir, pattern='.*Val.*csv$', full.names=TRUE)
trainPred <- list.files(predDir, pattern='.*Train.*csv$', full.names=TRUE)
testPred <- list.files(predDir, pattern='.*Test.*csv$', full.names=TRUE)

cvPreds <- read.csv(testPred, stringsAsFactors = F)
# hlist <- doSnrHist(cvPreds, by='pred')
allTrue <- purrr::reduce(doSnrHist(cvPreds, by='true'), `+`)
allTrue
ggsave(plot=allTrue, filename='SNRErr_CombOrig_ByTrue.png', width=18, height=12, units='in')
allPred <- purrr::reduce(doSnrHist(cvPreds, by='pred'), `+`)
allPred
ggsave(plot=allPred, filename='SNRErr_CombOrig_ByPred.png', width=18, height=12, units='in')

cvPreds %>%
    filter(true == 3, pred != 3) %>%
    filter(station == 'PASCAL_19_K_OE12') %>%
    ggplot() +
    geom_point(aes(x=p2, y=p3, col=snr), size=3) +
    geom_abline(slope=1, intercept=0) +
    geom_abline(slope=-1, intercept=1) +
    scale_color_gradientn(colors=viridisLite::viridis(25))
