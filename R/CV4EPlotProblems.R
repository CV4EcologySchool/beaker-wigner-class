ev <- df[c('station', 'eventId')]
ev$station <- gsub('^Station-([0-9|-]{1,5})_(Soundtrap|Card)-([A-z]{1,2}).*\\.(OE[0-9]{1,4})',
                   'PASCAL_\\1_\\3\\4', ev$eventId)
ev <- distinct(ev)

nev <- lapply(split(ev, ev$station), function(x) nrow(x))
badStation <- names(nev[nev>1])
ev[ev$station %in% badStation,]

gsub('^Station-([0-9|-]{1,5})_(Soundtrap|Card)-([A-z]{1,2}).*(Part[0-4]{1}){0,2}.*\\.(OE[0-9]{1,4})',
     'PASCAL_\\1_\\3\\4\\5',
     c("Station-1_Soundtrap-A_MASTER-BW_wGPS.OE19",'Station-24_Soundtrap-F_MASTER-BW 15dB Part2_wGPS.OE1'))

ev <- df
ev$station <- gsub('^Station-', '', ev$eventId)
ev$station <- gsub('(Soundtrap|Card)', '', ev$station)
ev$station <- gsub('MASTER-BW', '', ev$station)
ev$station <- gsub('wGPS', '', ev$station)
ev$station <- gsub('-', '', ev$station)
ev$station <- gsub('256kHzOnly', '', ev$station)
ev$station <- gsub('\\.', '_', ev$station)
ev$station <- gsub('(ETG)', '', ev$station)
ev$station <- gsub('15dB', '', ev$station)
ev$station <- gsub('Part', '_', ev$station)
ev$station <- gsub(' ', '', ev$station)
ev$station <- gsub('_{2,5}', '_', ev$station)
ev$station <- paste0('PASCAL_', ev$station)
ev$drift <- gsub('(PASCAL_[0-9]{1,4}).*', '\\1', ev$station)
length(unique(ev$station))
unique(ev$station)

library(ggplot2)
library(dplyr)
# detection plot
dplot <- ev %>%
    filter(species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')) %>%
    # group_by(drift, species) %>%
    # summarise(count=n()) %>%
    # str
    ggplot(aes(x=drift, fill=species)) +
    geom_bar()
# group plot
eplot <- ev %>%
    select(species, drift, station) %>%
    distinct() %>%
    filter(species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')) %>%
    # group_by(drift, species) %>%
    # summarise(count=n()) %>%
    # str
    ggplot(aes(x=drift, fill=species)) +
    geom_bar()
library(cowplot)
plot_grid(dplot, eplot, nrow=2)

ev %>%
    filter(species %in% c('ZC', 'BB', 'MS', 'BW43', 'BW37V')) %>%
    group_by(detectorName, species) %>%
    summarise(n())
ev %>%
    ggplot(aes(x=species, fill=detectorName)) +
    geom_bar()

ev %>%
    select(species, eventId) %>%
    distinct() %>%
    group_by(species) %>%
    summarise(n())


ev %>%
    select(species, drift, station) %>%
    distinct() %>%
    filter(species == 'BW43') %>%
    group_by(drift) %>%
    summarise(n())
