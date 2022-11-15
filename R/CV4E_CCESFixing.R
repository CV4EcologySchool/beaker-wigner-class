##### CCES CV4E Processing ####
driftDirs <- list.dirs('D:/CCES_2018/Finalized BW Analyses/', full.names = T, recursive = F)

checkSqlPoss <- function(d) {
    files <- list.files(d, pattern='sqlite3$', full.names = T, recursive = F)
    if(length(files) >= 1) {
        return(files)
    }
    dirs <- list.dirs(d, full.names = T, recursive = F)
    if(length(dirs) == 0) {
        return(NULL)
    }
    lapply(dirs, checkSqlPoss)
}

dbs <- lapply(driftDirs, function(d) {
    possible <- unlist(checkSqlPoss(d))
    final <- grepl('[Ff]inal', basename(possible))
    if(sum(final) == 1) {
        return(possible[final])
    }
    jst <- grepl('JST', basename(possible))
    if(sum(jst) == 1) {
        return(possible[jst])
    }
    warning('NO can ', d)
})
library(PAMpal)
dbs <- unlist(dbs)

# SM3M (4 and 17) had 10-50kHz filter not just 10kHz +
db <- c('D:/CCES_2018/Finalized BW Analyses/Drift-17 (completed by JST)/12 dB threshold 10 kHz filter/PamGuard64 2_00_16e Drift-17.sqlite3',
        'D:/CCES_2018/Finalized BW Analyses/Drift-4/12 dB threshold 10 kHz filter/PamGuard64 2_00_16e Drift-4b.sqlite3')
bin <- dirname(db)
library(PAMpal)
pps <- PAMpalSettings(db, bin, sr_hz='auto', filterfrom_khz=10, filterto_khz=NULL, winLen_sec=.0025)
ccesSM3M <- processPgDetections(pps, mode='db', id='CCES_SM3M')

# Some drifts are at 576kHz (15,16,18,19,20,21,22,23)
db <- dbs[c(4, 5, 7, 8, 9, 10, 11, 12)]
bin <- dirname(db)
pps <- PAMpalSettings(db, bin, sr_hz=288e3, filterfrom_khz=10, filterto_khz=NULL, winLen_sec=.0025)
cces576 <- processPgDetections(pps, mode='db', id='CCES_576')
cces576 <- setSpecies(cces576, method='pamguard')
cces576 <- addGps(cces576)
renamer <- data.frame(old=c('?BW', '?GG'), new=c('BWposs', 'GGposs'))
cces576 <- setSpecies(cces576, method='reassign', value=renamer)
for(e in seq_along(events(cces576))) {
    settings(cces576[[e]])$sr <- 288e3
}
saveRDS(cces576, 'ccesStudy576.rds')
ccesWig576 <- processAllWig(cces576, wl=128, sr=192e3, dir='../../CV4Ecology/CCES576_WL128', dataset='cces', fsmooth=NULL)

ev <- read.csv('../../CV4Ecology/CCES_WL128/labels/CCES_WL128_All.csv', stringsAsFactors = FALSE)
wigJoin <- select(ccesWig576, snr, wigMax, wigMin, file, sr)
wigJoin$file <- basename(wigJoin$file)
test <- bind_rows(lapply(split(ev, ev$sr), function(e) {
    if(e$sr[1] == 576e3) {
        e$snr <- NULL
        e$sr <- NULL
        e$wigMax <- NULL
        e$wigMin <- NULL
        e$fJoin <- basename(e$file)
        e <- left_join(e, wigJoin, by=c('fJoin'='file'))
        e$fJoin <- NULL
    }
    e
}))

ev %>%
    filter(species %in% c('ZC', 'BB', 'MS', 'BW37V', 'BW43')) %>%
    mutate(d = gsub('_[0-9]{1,2}', '', drift)) %>%
    filter(!d %in% c('4', '17')) %>%
    group_by(species, trainCat) %>%
    summarise(n=length(unique(station)))
