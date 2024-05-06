Pre-processing from "extractBrainSignalsSample.py", must reference raw data folder with data in format "S00-Name-flavor", and must be in same directory as "brain_signal_similarity.py"

Model generated using evaluateML.py, includes export command, PCA, Randomforest, and MUST USE pre-processed data file. 

I trained two models, unable to upload second, email me and I'll send it to you: 
1) "RF_classifier.jotlib" is trained using the processed "solo_data.csv" and only uses sugar and melon trials, which were performed independently. (Doesn't report to the server well)
2) "RF_all_classifier.jotlib" is trained using the processed "mixed_data.csv" which contains sugar and melon from independent trials AND salt, water, lemon from mixed trials.  

Make sure "brain_signal_similarity.py" and classifier are in the same directory as "server.py"

HTML template is "template.html" make sure it is in a separate folder labelled "templates" located within the same directory as "server.py"

1) Run server.py
2) Upload any signal file, click submit.
3) File is processed within server.py, top features are extracted, classified, and returned via server.
4) profit
