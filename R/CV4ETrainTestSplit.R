#CV4E Data splitting code and process
# helper for num in each species
library(dplyr)

checkNSpec <- function(x, sp) {
    sapply(sp, function(s) {
        sum(x$species == s)
    })
}
# do splitting into train/val/test based on probabilties n times
bwDataSplitter <- function(x, probs=c(.7, .15, .15), n=1, by=c('event', 'detection'), seed=112188) {
    set.seed(seed)
    switch(match.arg(by),
           'detection' = {
               splitCols <- c('eventId', 'drift', 'species', 'UID')
           },
           'event' = {
               splitCols <- c('eventId', 'drift', 'species')
           }
    )
    # make subset of x to do split and check against original dets
    x <- select(x, all_of(splitCols))
    x <- distinct(x)
    drifts <- unique(x$drift)
    result <- vector('list', length=n)
    pb <- txtProgressBar(min=0, max=n, style=3)
    for(i in 1:n) {
        samps <- sample(1:3, size=length(drifts), prob=probs, replace=TRUE)
        splitList <- list(train = x[x$drift %in% drifts[samps == 1], ],
                          val = x[x$drift %in% drifts[samps == 2], ],
                          test = x[x$drift %in% drifts[samps == 3], ])
        result[[i]]$split <- samps
        result[[i]]$species <- sapply(splitList, function(s) {
            checkNSpec(s, unique(x$species))
        })
        setTxtProgressBar(pb, i)
    }
    list(drifts=drifts, split=result)
}
# check splits against minimum values
isGoodSplit <- function(x, min=c(1, 1, 1)) {
    x <- x$species
    min(x[,1]) > min[1] &
        min(x[,2]) > min[2] &
        min(x[, 3]) > min[3]
}
# assign score to each split. Used 'prob' metric - compares distribution
# of events to desired distribution and squares difference.
splitScore <- function(x, probs=c(.7, .15, .15), mode=c('prob', 'diff')) {
    x <- x$species
    mode <- match.arg(mode)
    switch(mode,
           'diff' = {
               out <- apply(x[-1,], 1, FUN=function(s) {
                   sum(abs(sum(s) * probs - s))
               })
               sum(out^2)
           },
           'prob' = {
               out <- apply(x, 1, FUN=function(x) {
                   sum((x / sum(x) - probs)^2)
               })
               sum(out)
           })
}

writeTrTeCsv <- function(x, dir='.', name='PASCAL_BW',
                         cols=c('species', 'drift', 'station', 'file', 'wigMin','wigMax',
                                'dBPP', 'snr', 'Latitude','Longitude', 'ici', 'survey', 'sm3m')) {
    x <- select(x, all_of(c('trainCat', cols)))
    x$file <- gsub('../../CV4Ecology/', '', x$file)
    lapply(split(x, x$trainCat), function(s) {
        suff <- switch(as.character(s$trainCat[1]),
                       '1' = 'Train',
                       '2' = 'Val',
                       '3' = 'Test'
        )
        fname <- file.path(dir, paste0(name, '_', suff, '.csv'))
        write.csv(s[cols], file=fname, row.names=FALSE)
        fname
    })
}

# set seed before splitting for reproducibility
set.seed(112188)
# ev <- read.csv('../../CV4Ecology/PASCAL_BW_WL128.csv', stringsAsFactors = FALSE)
# ev <- read.csv('../../CV4Ecology/PASCAL_BW_WL128SNR_F33.csv', stringsAsFactors = FALSE)
ev <- read.csv('../../CV4Ecology/CCES_WL128/labels/CCES_WL128_All.csv', stringsAsFactors = FALSE)
combined <- read.csv('../../CV4Ecology/Combined_All.csv', stringsAsFactors = F)
combined$trainCat <- NULL
# evSp <- filter(ev, species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V'))
splits <- bwDataSplitter(dplyr::filter(combined, species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V'), !sm3m), n=1e5, by='det')
# saveRDS(splits, file='CCES_DetSplit.rds')
saveRDS(splits, file='Combined_DetSplit.rds')
# splits <- readRDS('../../CV4Ecology/splitByDet.rds')
splits <- readRDS('../../CV4Ecology/splitByEv.rds')
# first drop any with 0 of a species in any bin
good <- sapply(splits$split, isGoodSplit)
isGood <- data.frame(ix=which(good))
isGood$scoreProb <- sapply(splits$split[good], function(x) splitScore(x, probs=c(.7, .15, .15), mode='prob'))
isGood <- arrange(isGood, scoreProb)
possBest <- isGood$ix[1:5]
splits$split[[possBest[1]]]$species
drJoin <- data.frame(drift=splits$drifts,
                     trainCat=splits$split[[possBest[1]]]$split)
combined <- left_join(combined, drJoin)
ev$trainCat[is.na(ev$trainCat)] <- 1
csvs <- writeTrTeCsv(combined, dir='../../CV4Ecology/Combined', name='Combined_All10k')
writeTrTeCsv(dplyr::filter(combined, species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')), dir='../../CV4Ecology/Combined', name='Combined_BW10k')

# plottys
library(ggplot2)
plotEv <- left_join(evSp, drJoin)
combined %>%
    group_by(drift) %>%
    summarise(plotLat = mean(Latitude),
              plotLon = mean(Longitude),
              # drift=as.character(unique(drift))) %>%
              trainCat = mean(trainCat),
              survey=unique(survey)) %>%
    ggplot() +
    geom_point(aes(x=plotLon, y=plotLat, col=as.character(trainCat)), size=5) +
    facet_wrap(~survey)
    # geom_point(aes(x=plotLon, y=plotLat, col=drift), size=5)

ev %>%
    select(drift, species, station, trainCat) %>%
    dplyr::filter(species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')) %>%
    distinct() %>%
    ggplot() +
    geom_bar(aes(x=drift, fill=species))

ev %>%
# pascal %>%
    dplyr::filter(species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')) %>%
    # dplyr::filter(species %in% c('BW37V')) %>%
    # filter(!sm3m) %>%
    group_by(drift) %>%
    summarise(start=min(UTC), end=max(UTC), length=difftime(max(UTC), min(UTC), units='days'))
## SPLIT BY WEEK ##
ev <- ev %>%
    group_by(drift) %>%
    mutate(nDays = as.numeric(difftime(UTC, min(UTC), units='days'))) %>%
    ungroup()

ev$drift <- paste0(ev$drift, '_', floor(ev$nDays/7.404))
length(unique(ev$drift))
ev$drift[ev$station == 'CCES_8_OE57'] <- '8_3'
ev$nDays <- NULL
ev$drift_w <- NULL
evDrift <- ev %>%
    group_by(station) %>%
    summarise(n=length(unique(drift)))
evDrift[evDrift$n > 1,]

#### SP distribution by drift ####
spByDriftPlot <- function(x) {
    x %>%
        filter(species %in% c('ZC', 'MS', 'BB', 'BW43', 'BW37V')) %>%
        select(drift, species, station) %>%
        distinct() %>%
        ggplot() +
        geom_bar(aes(x=drift, fill=species)) +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
}
library(patchwork)
(spByDriftPlot(pascal) + spByDriftPlot(filter(pascal, !sm3m))) /
    (spByDriftPlot(cces) + spByDriftPlot(filter(cces, !sm3m)))