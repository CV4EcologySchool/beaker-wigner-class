require(reticulate)
require(PAMmisc)
# require(signal)
library(PAMpal)

# doWigBin <- function(x, wl=128, dir='.', name='', channel=1, sr=192e3, pal) {
#     if(x$sr < sr) {
#         return(NA)
#     }
#     clip <- downSamp(x$wave[, channel], x$sr, sr)
#     clip <- PAMpal:::clipAroundPeak(clip, length=wl)
#     # DO DOWNSAMP AND FILTERING
#     wig <- PAMmisc::wignerTransform(clip, n=wl, sr=sr, plot=FALSE)
#     filename <- file.path(dir, paste0(name, x$UID, '_C', channel, '.png'))
#     # png(filename = filename, width=wl, height=wl, units='px')
#     # par(mar=c(0,0,0,0))
#     # x11()
#     # image(t(wig$tfr), col=viridisLite::viridis(25), useRaster=FALSE)
#     # dev.off()
#     # wig <- wig$tfr
#     # wig$tfr <- (wig$tfr - min(wig$tfr)) / diff(range(wig$tfr))
#     # wig <- array(t(col2rgb(pal(wig$tfr))), dim=c(dim(wig$tfr), 3)) / 255
#     # # magRgb <- image_read(wig/255)
#     # # image_flip(magRgb)
#     #
#     # # wig <- array(wig$tfr, dim=c(dim(wig$tfr), 1))
#     # mag <- magick::image_read(wig)
#     # mag <- magick::image_flip(mag)
#     # magick::image_write(mag, path=filename)
#     # dev.off()
#     filename
# }

downSamp <- function(x, srFrom=NULL, srTo, filter=NULL) {
    if(!is.null(filter)) {
        x <- seewave::bwfilter(x, f=srFrom, from=filter * 1e3, n=4)
    }
    if(is.null(srFrom)) {
        srFrom <- x$sr
    }
    if(srFrom == srTo) {
        return(x)
    }
    y <- signal::filtfilt(signal::cheby1(8, 0.05, 0.8/(srFrom/srTo)), x)
    y[seq(1, length(x), length = srTo * length(x)/srFrom)]
}

procWigEvent <- function(ev, wl=128, dir='Writest', sr=192e3, np=NULL, dataset='bw', fsmooth=NULL, filter=NULL) {
    click <- getClickData(ev)
    click <- distinct(select(click, -detectorName))
    # click$station <- gsub('\\.OE[0-9]{1,4}', '', click$eventId)
    switch(dataset,
           'bw' = {
               click$station <- gsub('^Station-', '', click$eventId)
               click$station <- gsub('(Soundtrap|Card)', '', click$station)
               click$station <- gsub('MASTER-BW', '', click$station)
               click$station <- gsub('wGPS', '', click$station)
               click$station <- gsub('-', '', click$station)
               click$station <- gsub('256kHzOnly', '', click$station)
               click$station <- gsub('\\.', '_', click$station)
               click$station <- gsub('(ETG)', '', click$station)
               click$station <- gsub('15dB', '', click$station)
               click$station <- gsub('Part[0-9]{0,1}', '_', click$station)
               click$station <- gsub(' ', '', click$station)
               click$station <- gsub('_{2,5}', '_', click$station)
               click$station <- paste0('PASCAL_', click$station)
               click$drift <- gsub('(PASCAL_[0-9]{1,4}).*', '\\1', click$station)
               # click$station <- gsub('^Station-([0-9|-]{1,5})_(Soundtrap|Card)-([A-z]{1,2}).*\\.(OE[0-9]{1,4})',
               #                       'PASCAL_\\1_\\3_\\4', click$eventId)
           },
           'nbhf' = {
               click$station <- gsub('^NBHF Station-([0-9|-]{1,6}) ST4300-([A-z]).*\\.(OE[0-9]{1,4})',
                                     'PASCAL_NBHF_\\1_\\2_\\3', click$eventId)
           },
           'cces'= {
               click$station <- gsub('.* Drift-([0-9]{1,2}).*\\.(OE[0-9]{1,4})', 'CCES_\\1_\\2', click$eventId)
               click$drift <- gsub('CCES_([0-9]{1,2})_OE[0-9]{1,4}', '\\1', click$station)
           },
           'finclick' = {
               # click$station <- gsub('^CCES_2018_(DRIFT[0-9]{1,2})_.*', '\\1', click$eventId)
               click$station <- gsub('.*\\.([0-9]{12})\\.wav', '\\1', click$eventId)
               click$drift <- gsub('CCES_2018_(DRIFT[0-9]{1,2})_.*', '\\1', basename(click$db))
           },
           {
               click$station <- click$eventId
               click$drift <- click$station
           }
    )
    bin <- getBinaryData(ev, UID = click$UID)
    if(length(bin) < length(unique(click$UID))) {
        cat('\nOnly', length(bin), 'in event', id(ev), 'expected', length(unique(click$UID)),'\n')
        return(NULL)
        # missDets <- c(missDets, e)
        # setTxtProgressBar(pb, value=e)
        # next
    }
    # vScale <- scales::col_numeric(palette=scales::viridis_pal()(25), domain=c(0,1))
    bin <- bin[unique(click$UID)]
    if(is.null(np)) {
        np <- import('numpy')
    }
    wigOut <- wigNpArray(bin, n=wl, t=wl, dir=dir, filename=click$station[1], np=np, fsmooth=fsmooth, filter=filter, sr=sr)
    # click$filename <- wigOut$filename
    # click$wigMax <- wigOut$wigMax
    # click$wigUID <- wigOut$UID
    click <- cbind(click, wigOut)
    click
}

wigPreproc <- function(x, wl=128, srFrom, srTo=192e3, c=1, filter=NULL) {
    if(is.list(x)) {
        srFrom <- x$sr
        x <- x$wave[,c]
    }
    x <- downSamp(x, srFrom, srTo, filter)
    x <- PAMpal:::clipAroundPeak(x, wl)
    x
}

wigNpArray <- function(data, n=128, t=128, dir='.', filename, np, sr=192e3, fsmooth=NULL, filter=NULL) {
    nWavs <- sum(sapply(data, function(x) ncol(x$wave)))
    result <- array(NA, dim = c(n, t, nWavs))
    isNA <- integer(0)
    wigMax <- numeric(0)
    wigMin <- numeric(0)
    dataSr <- numeric(0)
    files <- character(0)
    snr <- numeric(0)
    for(d in seq_along(data)) {
        nChan <- ncol(data[[d]]$wave)
        dataSr <- c(dataSr, rep(data[[d]]$sr, nChan))
        if(is.null(data[[d]]$wave) ||
           data[[d]]$sr < sr) {
            isNA <- c(isNA, nChan*(d-1) + 1:nChan)

            wigMax <- c(wigMax, rep(NA, nChan))
            wigMin <- c(wigMin, rep(NA, nChan))
            snr <- c(snr, rep(NA, nChan))
            files <- c(files, rep(NA, nChan))
            next
        }
        for(c in 1:nChan) {

            thisWav <- downSamp(data[[d]]$wave[, c], data[[d]]$sr, sr, filter=filter)
            noise <- PAMpal:::clipAroundPeak(thisWav, t, TRUE)
            thisWav <- PAMpal:::clipAroundPeak(thisWav, t, FALSE)
            # do snr

            # not sqrt because we are just logging below
            sigRMS <- 10*log10(mean(thisWav^2))
            noiseRMS <- 10*log10(mean(noise^2))
            snr <- c(snr, sigRMS - noiseRMS)

            # end snr
            if(is.null(fsmooth)) {
                wig <- PAMmisc::wignerTransform(thisWav, n=n, sr=sr)$tfr
            } else {
                wig <- smoothWVD(thisWav, n=n, sr=sr, fw=fsmooth)$tfr
            }
            # wig <- (wig - min(wig)) / (max(wig) - min(wig)) * intMax
            wMin <- min(wig)
            wMax <- max(wig)
            max <- max(abs(c(wMin, wMax)))
            wig <- wig / max * 127.5 + 127.5
            wig <- np_array(wig, dtype='uint8')
            fname <- paste0(filename, '_', data[[d]]$UID, '_C', c)
            # browser()
            np$save(file.path(dir, fname), wig)
            files <- c(files, paste0(file.path(dir, fname), '.npy'))
            # result[,,2*(d-1) + c] <- wig
            wigMax <- c(wigMax, wMax)
            wigMin <- c(wigMin, wMin)
        }
    }
    UID <- rep(names(data), each=nChan)
    if(length(isNA) > 0) {
        result <- result[,,-isNA]
        # UID <- UID[-isNA]
        # UID[isNA] <- NA
        # wigMax <- wigMax[-isNA]
    }
    if(dim(result)[3] == 0) {
        return(list(file=NA,
                    wigUID=NA,
                    wigMax=NA,
                    wigMin=NA,
                    snr=NA,
                    sr=NA))
    }
    # UID <- unique(UID)
    # result <- (result - min(result)) / (max(result)-min(result)) * intMax
    # result <- np_array(result, dtype='float')
    # result <- np_array(result, dtype='int8')
    # np$savez_compressed(file.path(dir, filename), result)
    # list(file=paste0(file.path(dir, filename), '.npz'),
    list(file = files,
         wigUID = UID,
         wigMax=wigMax,
         wigMin=wigMin,
         snr=snr,
         sr=dataSr)
    # result
}

processAllWig <- function(x, wl=128, sr=192e3, dir='NPWig', dataset=c('bw', 'nbhf', 'cces', 'finclick'), fsmooth=NULL, filter=NULL) {
    start <- Sys.time()
    on.exit(cat('\nTook', as.character(round(difftime(Sys.time(), start, units='mins'), 1)), 'minutes'))
    if(!dir.exists(dir)) {
        dir.create(dir)
    }
    dataset <- match.arg(dataset)
    # use_python('../../CV4Ecology/venv/Scripts/python.exe')
    # use_virtualenv('../../CV4Ecology/venv')
    np <- import('numpy')
    result <- vector('list', length=length(events(x)))
    cat('Writing NPArrays...\n')
    pb <- txtProgressBar(min=0, max=length(result), style=3)
    for(i in seq_along(result)) {
        if(!is.na(species(x[[i]])$id)) {
            thisDir <- file.path(dir, species(x[[i]])$id)
        } else {
            thisDir <- dir
        }
        if(!dir.exists(thisDir)) {
            dir.create(thisDir)
        }
        result[[i]] <- procWigEvent(x[[i]], wl=wl, sr=sr, dir=thisDir, np=np, dataset=dataset, fsmooth=fsmooth, filter=filter)
        setTxtProgressBar(pb, value=i)
    }
    bind_rows(result)
}

smoothWVD <- function(signal, n=NULL, sr, plot=FALSE, tw=NULL, fw=NULL) {
    if(inherits(signal, 'Wave')) {
        sr <- signal@samp.rate
        signal <- signal@left / 2^(signal@bit - 1)
    }
    if(inherits(signal, 'WaveMC')) {
        sr <- signal@samp.rate
        signal <- signal@.Data[, 1] / 2^(signal@bit - 1)
    }
    if(is.null(tw)) {
        tw <- n %/% 10
    }
    tw <- tw + 1 - (tw %% 2)
    tw <- signal::hamming(tw)
    if(is.null(fw)) {
        fw <- n %/% 4
    }
    fw <- fw + 1 - (fw %% 2)
    fw <- signal::hamming(fw)
    tmid <- (length(tw)-1) %/% 2
    fmid <- (length(fw)-1) %/% 2

    analytic <- PAMmisc:::toAnalytic(signal)[1:length(signal)] # size changed during toAnalytic function
    conjAnalytic <- Conj(analytic)
    if(is.null(n)) {
        n <- PAMmisc:::nextExp2(length(analytic))
    }

    nRow <- n # nFreq bins
    nCol <- length(analytic) # nTimesteps
    # nCol <- n # nTimesteps

    tfr <- matrix(0, nRow, nCol)

    for(iCol in 1:nCol) {
        taumax <- min(iCol-1, nCol-iCol, round(nRow/2)-1, fmid)
        # cat('Max', iCol + taumax,
        #     'Min', iCol-taumax, '\n')
        tau <- -taumax:taumax
        indices <- (nRow + tau) %% nRow + 1
        # * .5 in PG?
        # print(tau+fmid)
        # browser()
        tfr[indices, iCol] <- fw[tau + fmid+1] * analytic[iCol+tau] * conjAnalytic[iCol-tau] / 2

        tau <- round(nRow/2)
        if(iCol + tau <= nCol &&
           iCol - tau >= 1) {
            # browser()
            # PG is like this, wv.wge is just the same fucking thing???
            tfr[tau+1, iCol] <- fw[tau+fmid+1]*(analytic[iCol+tau] * conjAnalytic[iCol-tau] +
                                                    analytic[iCol-tau] * conjAnalytic[iCol+tau])/2
        }
    }
    tfr <- apply(tfr, 2, fft)
    result <- list(tfr=Re(tfr), t=1:nCol/sr, f=sr/2*1:nRow/nRow)
    if(plot) {
        image(t(result$tfr), xaxt='n', yaxt='n',
              ylab='Frequency (kHz)', xlab = 'Time (ms)',
              col = viridisLite::viridis(25), useRaster=TRUE)
        xPretty <- pretty(result$t, n=5)
        # axis(1, at = 1:4/4, labels = round(1e3*max(result$t)*1:4/4, 3))
        axis(1, at=xPretty / max(result$t), labels=xPretty*1e3)
        yPretty <- pretty(result$f, n=5)
        # axis(2, at = 1:4/4, labels = round(max(result$f)*1:4/4/1e3, 1))
        axis(2, at = yPretty / max(result$f), labels=yPretty/1e3)
    }
    result
}