# processing CV4E data
library(PAMpal)

# Pascal Beakers - DBs are not organized neatly, gotta dig into file structure for this shit
driftFolders <- list.dirs('D:/PASCAL_2016/Binaries&DBs by Drift ReRun w 2_00_16e/', recursive = FALSE, full.names = TRUE)
allDb <- unlist(sapply(driftFolders, function(x) {
    dirs <- list.dirs(x, recursive = FALSE, full.names = TRUE)
    if(any(grepl('Binaries wUID', dirs))) {
        db <- list.files(x, pattern='GPS.sqlite3', full.names=TRUE)
        if(length(db) == 0) {
            cat('\nNo db found in folder', x)
        }
        return(db)
    }
    sapply(dirs, function(y) {
        db <- list.files(y, pattern='GPS.sqlite3', full.names=TRUE)
        if(length(db) == 0) {
            cat('\nNo db foundin folderY', y)
        }
        db
    })
})) %>% unname()

allPps <- PAMpalSettings(db=allDb[1:50], binaries = dirname(allDb)[1:50], sr_hz='auto', filterfrom_khz=10, filterto_khz=NULL, winLen_sec=.0025)

bwData <- processPgDetections(allPps, mode='db', id='PASCAL_BW')
bwData <- setSpecies(bwData, method='pamguard')
bwData <- addGps(bwData)
gps <- gps(bwData)
View(filter(gps, grepl('Station-24', db)))
species(bwData)
saveRDS(bwData, file='PASCAL_Base_AcSt.rds')
bwData <-  readRDS('../../CV4Ecology/PASCAL_Base_AcSt.rds')
table(species(bwData))
getWarnings(bwData)$message[3]
speciesRename <- data.frame(old = c('BW34-50', 'BW46', 'BW50-75', '?BW', 'Zc', 'Pm', 'SW', '?Pm', 'Oo', '?NBHF', '?PM', 'BW?'),
                            new = c('BW37V', 'MS', 'MS', 'BWunid', 'ZC', 'PM', 'PM', '?PM', 'OO', 'NBHF', 'PM', 'BW'))
bwData <- setSpecies(bwData, method='reassign', value=speciesRename)
# Pascal NBHF - normally organized, had to convert from 1.15 to 2.0+
db <- list.files('D:/PASCAL_2016/PASCAL NBHF UID/Databases/', full.names=TRUE)
bin <- 'D:/PASCAL_2016/PASCAL NBHF UID/Binaries/'

pps <- PAMpalSettings(db, bin, sr_hz='auto', filterfrom_khz=10, filterto_khz=NULL, winLen_sec=.0025)
nbhfData <- processPgDetections(pps, mode='db', id='PASCAL_NBHF')
nbhfData <- setSpecies(nbhfData, method='pamguard')
saveRDS(nbhfData, file='PASCAL_NBHF_AcSt.rds')
nbhfData <- readRDS('PASCAL_NBHF_AcSt.rds')
# PASCAL NBHF NOTES
# station 7 no detections, station 16 no detections, 28 none.
# 7, 12, 15, 16, 25, 26-29, 28 no detections after processing

bw <- getClickData(bwData)
bw$db <- gsub('\\.OE[0-9]{1,4}', '', bw$eventId)
bw$station <- gsub('^Station-([0-9|-]{1,5})_(Soundtrap|Card)-([A-z]{1}).*', 'PASCAL_\\1_\\3', bw$db)

nbhf <- getClickData(nbhfData)
# nbhf$db <- gsub('\\.OE[0-9]{1,4}', '', nbhf$eventId)
nbhf$station <- gsub('^NBHF Station-([0-9|-]{1,6}) ST4300-([A-z]).*\\.(OE[0-9]{1,4})', 'PASCAL_NBHF_\\1_\\2_\\3', nbhf$eventId)



library(profvis)
system.time(procWigEvent(bwData[[165]]))
profvis(procWigEvent(bwData[[165]]))

pascalWig <- processAllWig(bwData, wl=128, sr=192e3, dir='../../CV4Ecology/WL128SNR_10k', dataset='bw', filter=10)
# write.csv(allWigNp, row.names = F, file='../../CV4Ecology/PASCAL_BW_WL128SNR_F33.csv')
# hm... there are some events with partial SR. porbably need to not write? think more
bigWig <- which(allWigNp$wigMax < .0000001)
bigBin <- getBinaryData(bwData, UID=allWigNp$UID[allWigNp$station == 'PASCAL_10_E_OE1'])
clip <- downSamp(bigBin[[5]]$wave[, 1], srFrom = bigBin[[1]]$sr, 192e3)
clip <- PAMpal:::clipAroundPeak(clip, 128)
plot(clip, type='l')
wiggy <- wignerTransform(clip, n=128, sr=bigBin[[1]]$sr, plot=T)
plot(wiggy$tfr[1,], type='l')



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
bins <- dirname(dbs)
#### drift 4 and 17 have a fucked 10-50k filter, others 10 ####
pps <- PAMpalSettings(db=dbs, binaries = bins, sr_hz='auto', filterfrom_khz=10, filterto_khz=NULL, winLen_sec=.0025)
ccesData <- processPgDetections(pps, mode='db', id='CCES_BW')
ccesData <- setSpecies(ccesData, 'pamguard')
table(species(ccesData))
renamer <- data.frame(old=c('?BW', '?GG'), new=c('BWposs', 'GGposs'))
ccesData <- setSpecies(ccesData, method='reassign', value=renamer)
table(species(ccesData))
ccesData <- addGps(ccesData)
saveRDS(ccesData, file='ccesStudy.rds')

ccesWig <- processAllWig(ccesData, wl=128, sr=192e3, dir='../../CV4Ecology/CCES_WL128', dataset='cces', fsmooth=NULL)
write.csv(ccesWig, file = '../../CV4Ecology/CCES_WL128_All.csv', row.names = FALSE)

#### Dropping SM3M, add ICI, combine   ####
pascal <- read.csv('../../CV4Ecology/PASCAL_BW_WL128SNR.csv', stringsAsFactors = F)
pascal$sm3m <- pascal$drift %in% paste0('PASCAL_', c('7', '10', '13', '17'))
pstudy <- readRDS('../../CV4Ecology/PASCAL_Base_AcSt.rds')
pstudy <- calculateICI(pstudy, time='peakTime')
ici <- getICI(pstudy)
ici <- bind_rows(lapply(ici, function(x) x['All_ici']), .id='eventId')
ici <- rename(ici, ici = All_ici)
pascal <- left_join(pascal, ici)
# too few clicks in event, Anne recs switch to BW fro BW43 bc not confident
pascal$species[pascal$station == 'PASCAL_19_K_OE15'] <- 'BW'

cces <- read.csv('../../CV4Ecology/CCES_WL128_All.csv', stringsAsFactors = F)
cces$drift
cstudy <- readRDS('ccesStudy.rds')
cstudy <- calculateICI(cstudy, time='peakTime')
ici <- getICI(cstudy)
ici <- bind_rows(lapply(ici, function(x) x['All_ici']), .id='eventId')
ici <- rename(ici, ici = All_ici)
cces <- left_join(cces, ici)
cces$sm3m <- grepl('^4|^17', cces$drift)
cces$drift <- gsub('_[0-9]{1,2}', '', cces$drift)
cces$drift <- paste0('CCES_', cces$drift)
pascal$detectorName <- NULL
pascal$wigMin <- NA
pascal$survey <- 'pascal'
cces$survey <- 'cces'
combined <- rbind(pascal, cces)

