Pre-processing from "extractBrainSignalsSample.py", must reference folder with data in format "S00-Name-flavor"

Model generated using evaluateML.py, includes export command, and MUST USE pre-processed data file. 

In this case it is trained using "mixed_data.csv", which contains sugar and melon from independent trials and salt, water, lemon from mixed trials. 

Model exported as RF_all_classifier.jotlib

Make sure "brain_signal_similarity.py" and "RF_all_classifier.jotlib" are in the same directory as "server.py"

HTML template is "template.html" make sure it is in a separate folder labelled "templates" located within the same directory as "server.py"

1) Run server.py
2) Upload any signal file, click submit.
3) File is processed within server.py, top features are extracted, classified, and returned via server.
4) profit
