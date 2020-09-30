To run, just call:

python perceptron.py

By default, runs online, average, and all kernel perceptrons, and outputs images and files. Can also test one part at a time by doing:

python perceptron.py online
python perceptron.py average
python perceptron.py kernel
python perceptron.py kernel 3

Confirmed to be running on Python 3.6.3, pandas version 0.20.3. Note that this will NOT work on any version of Pandas past 0.21.0 due to the deprecation of pandas.Series.argmax.
