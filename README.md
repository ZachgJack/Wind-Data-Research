# Wind-Data-Research

The MLPlotter.py file is used as the controll for LSTM_Prototype.py, K_means_Plotter, and soon-to-be more files.

UPDATE 6/7/2025:
MLPlotter.py (and by extension LSTM_Prototype) should be able to take any csv file, sift through the data and insert it into the model without issue, as long as the file has a column that is labled "Wind Speed" and one that is labled "Date time". Unfortunately I have only been able to test Akron Fulton International Airport.csv and Cleveland Weather Data 2013-2022 10min.csv so far. I will look into finding more data sets online.
Added a portion that makes the hours and day of the year cyclical, so that 23:00 is next to 00:00.

Started work on a K-means plotter to see if any patterns can be gleaned from that.
