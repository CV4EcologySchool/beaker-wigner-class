
## TODO:
    Should I be adding hard negatives
    Can I enforce y-axis meaning or is it just going to learn it
    Implementing early stopping?
    Implement focal loss for multiclass
    https://medium.com/swlh/multi-class-classification-with-focal-loss-for-imbalanced-datasets-c478700e65f5
    https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872
    Curriculum learning? Idk if our problem is necessarily class imbalance actually
    Using SNR as flag to mark "ones to check" for people
    Using SNR to scale predicted probabilities??? Hmmm. I could do something
        like create a precision-recall curve based on "threshold" or whatever
        way I'm scaling by SNR and see how much this affects prediction of event
    
    Average spectrum?
    
    Should we try this with detection-level ICI? Plots suggest that overall these should
        be well-distributed around the expected levels. Event level is a little odd because
        for ones with many clicks you are giving a huge signal at a single value - this 
        value may be biased by other interfering signals. Within event there *should* be
        a lot at the correct-ish value. 
    RPY2 LETS ME DO R STUFF IN PYTHON? GGPLOT BROS FOR LIFE

## TODONE:    
    oversampling lower rep classes
    Evaluate by event not by detection
    Can we incorporate enviro data to each detection
    Change the damn weights to stop warning deprecated
    Can slap stuff onto the 2nd to last layer as new features
    
    Augs to try - shift + fill shifted with grey/black
        blur to maybe reduce noise ones and improve high q ones
        T.RandomAffine has x-y amts
    Really want to look at the thing where it shows you which parts of image make
        it think class A vs B i forget the name - SALIENCY MAP IS DONE
        Gradient - https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
        DFF and GradCAM - https://github.com/jacobgil/pytorch-grad-cam
    Look at weights of ICI model to see if it is actually utilizing ? Also try mult 10
        - MULT 10 IS GREAT
    Try majority vote event prediction instead of average prob - NO REAL DIFF
    Try training ZC vs BB model - SAME AS ALL SPECIES
    Try to compare just ICI classification - how do we break up by only ICI and compare
        accuracy that way. Logit regression. - BAD CUZ BB COVERS ALL - GETS CONFUSED
    AvgPrec does not seem to be telling the full story - worse on ICI 10 model by
         a good chunk, but performance is much better at event level (from 46->69%
         on bad BB events, coming from 17->28% at detection level). F1 score is better
         for ICI model, so that is maybe a better metric or something at event level?   
## NOTES:
    BW43 event Pascal_19_OE15 should maybe switch to BW - too few clicks,
        Anne not super confident
    BW43 event Pascal_19_OE12 definitely good as BW43
    PASCAL_15_OE7 is MS **NOT** ZC as labeled! Model wins!!!
## ELI QUESTIONS:
    Updates - tried combining, lots of data issues - applying filters to make
        same same, removing some sites on bad noisy recorders. Tried saliency
        stuff, seems like it is looking at background for some lower freq calls.
        Okay bc that is same as "call is low freq" if lots of upper nothing. 
    Generally, not sure what to try next. Both times model has performed well
        except for one class. Nothing Ive tried seens to make much difference
        on that problem class.
    Why did ICI do nothing the way I incorporated it
    How do I know when to stop training
    How do I do this forever - If I want to do this as a job, do I need to go to school
    
## TRY FROM TALK
    ADding gaussian noise might be appropriate. https://github.com/pytorch/vision/issues/6192
    
    "Open Set" classification for future application
    "Classifier with reject option"
    
    WHEN ASKED ABOUT JOBS LINKS
    Sam Kelly
    Conservation AI slack channel
    https://www.climatechange.ai/
    https://www.microsoft.com/en-us/ai/ai-for-earth
    https://conservationxlabs.com/
    https://conservationtech.directory/