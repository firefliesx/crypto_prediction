# crypto_prediction
Please create a data repository and split the data by currency for both training and test data and make sure that the column headers exist in each currency data file as follows:
/crypto_prediction/data (master)
$ ls -ltrh
total 726M
-rw-r--r-- 1 danis 197609 167M Mar 28 14:49 BTCUSDT_training.csv
-rw-r--r-- 1 danis 197609 158M Mar 28 14:52 ETHUSDT_training.csv
-rw-r--r-- 1 danis 197609 149M Mar 28 14:53 LTCUSDT_training.csv
-rw-r--r-- 1 danis 197609 155M Mar 28 14:53 XRPUSDT_training.csv
-rw-r--r-- 1 danis 197609  27M Mar 28 14:53 BTCUSDT_test.csv
-rw-r--r-- 1 danis 197609  25M Mar 28 14:54 ETHUSDT_test.csv
-rw-r--r-- 1 danis 197609  24M Mar 28 14:54 LTCUSDT_test.csv
-rw-r--r-- 1 danis 197609  24M Mar 28 14:54 XRPUSDT_test.csv

You can use grep to do the splitting e.g.: `cat training.csv | grep BTCUSDT > BTCUSDT_training.csv` and make sure that you copy the column names from the original training.csv and prepend it on each training data file. Do the same for test data.
