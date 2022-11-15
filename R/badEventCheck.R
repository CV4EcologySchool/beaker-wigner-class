# cces preds
library(PAMpal)
library(dplyr)
cpred <- read.csv('../../CV4Ecology/export_cces_norm.5.04_snrfilt5/CCES_BW_Val_BadCCESpred.csv', stringsAsFactors = F)
ccesData <- readRDS('ccesStudy.rds')
cpred <- arrange(cpred, desc(p3))
# get UID from file names
predToUID <- function(x) {
    uid <- vector('character', length=nrow(x))
    for(i in seq_along(uid)) {
        uid[i] <- gsub(paste0(x$station[i], '_'), '', basename(x$file[i]))
        uid[i] <- gsub('_C[12].npy', '', uid[i])
    }
    uid
}
# wig max values
ev %>%
    filter(species %in% c('ZC', 'MS', 'BB', 'BW43', 'BW37V')) %>%
    filter(wigMax < .0005) %>%
    ggplot() +
    geom_density(aes(x=wigMax), col='blue') +
    geom_density(aes(x=abs(wigMin)), col='red')

# PASCAL sm3m drifts are 7, 10, 13, 17
# BAD BW43 EVENTS ARE:
# Fuckin station 19 only 4kHz filter
bad43 <- c('Station-19_Soundtrap-K_MASTER-BW_wGPS.OE12',
           'Station-19_Soundtrap-K_MASTER-BW_wGPS.OE15')
pascal <- readRDS('../../CV4Ecology/PASCAL_Base_AcSt.rds')
p43 <- filter(pascal, species == 'BW43')
avList <- vector('list', length=length(events(p43)))
for(e in seq_along(avList)) {
    avg <- calculateAverageSpectra(pascal[e], evNum=1, wl=128, plot=c(F,F), norm=F, filterfrom_khz)
    avList[[e]] <- avg$avgSpec
}
for(e in seq_along(avList)) {
    if(e == 1) {
        plot(x=avg$freq, y=avList[[e]], type='l')
    } else{
        lines(x=avg$freq, y=avList[[e]])
    }
}

av <- calculateAverageSpectra(p43, 5:37, wl=128, noise=T, filterfrom_khz = 10)

clk <- getClickData(p43[bad43])
bin <- getBinaryData(p43[bad43], clk$UID)
ix <- 357
clip <- PAMpal:::clipAroundPeak(bin[[ix]]$wave[,1], 128)
plot(clip, type='l')
wt <- PAMmisc::wignerTransform(clip, n=128, sr=bin[[ix]]$sr, plot=T)

goodSp <- c('ZC', 'BB', 'MS', 'BW43', 'BW37V')
# checking on shit
combined %>%
    filter(!sm3m) %>%
    filter(species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')) %>%
    select(drift, species, trainCat, eventId) %>%
    distinct() %>%
    filter(trainCat == 2) %>%
    with(table(species, drift))

# ici plot
combined %>%
    filter(species %in% goodSp) %>%
    select(species, ici, station, trainCat, survey) %>%
    distinct() %>%
    ggplot() +
    geom_density(aes(x=ici, col=species)) +
    xlim(0, .7)
    # geom_vline(xintercept = c(.09, .14, .22)
