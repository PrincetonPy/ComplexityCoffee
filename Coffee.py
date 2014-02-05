# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# The Scipy Stack - A Hands-On Session

# <markdowncell>

# **Coffee Consumption Time Series**
# 
# Quentin CAUDRON
# 
# This data describes the consumption of coffee at the espresso machine in the Centre for Complexity Science, at the University of Warwick. The data collection exercise was facilitated by an Android app which automatically timestamped incoming data, and added the submitting user's name. *In theory*, whenever contributors ( thanks, guys ! ) had a coffee, they would query the machine as to how many coffees it had made to date, and submit the count via the Android app. *In practice*, the small number of contributers means that the physical process is poorly and irregularly sampled, full of holes, and not of the best quality. Concisely put, this is **real data**, my worst nightmare. Hey, at least it's strictly monotonic...
# 
# The purpose of this notebook is to perform some basic time series analysis on the data using primarily the functionality of Numpy, Scipy, and Matplotlib. A few other cool things might be thrown in for fun.

# <headingcell level=3>

# Exercises

# <markdowncell>

# A list of things to try :
# 
# **Reading the Data**
# 
# FASTEST : Using Pickle
# 
# - For speed, I've got a Pickle file ready to be imported. This is for today's session, and, of course, won't be available when you want to actually import your own data. Still, for speed today, if you just want access to the data, call :
# 
# `with open("pickled_coffee.p") as f :
#     times, coffees, names, day_of_week, time_seconds, hour_of_day = pickle.load(f)`
#     
# Note the indent ! The variables are now each column of the .csv. In the code that follows, you'll need to replace the Pandas object with the relevant variable.
# 
# BEST : Using Pandas
# 
# - Use `read_csv(filename)`, where `filename` is a string. This will generate a Pandas *DataFrame* object
# - You can access DataFrame columns by using ["Column_Name"] after the object
# 
# FOR MASOCHISTS : Using native Python, one method
# 
# - Open the file using `myfile = open(filename)`
# - Iterate over the file using `for line in myfile`, which will give you a string
# - Strings have a handy `.split(",")` method, which splits the line at a comma
# - You'll need to ignore the first line, which contains headers
# 
# ALSO MASOCHISTIC : Using native Python, another method :
# 
# - Use Numpy's `genfromtxt` method to read the whole file into a variable all at once
# - There's a handy `skip_header` argument to lose the row of headers
# - You should also specify the delimiter
# - Finally, enforce `dtype = None` to get Python to guess the type of variable per column
# - The result is a list of tuples, not a list of lists
# 
# **Plotting the Data**
# 
# - After `import matplotlib.pyplot as plt`, you can call `plt.plot(x, y)` to plot a basic graph
# - You can use the column entitled "Time_Seconds" if you want easy access to the time variable
# - Otherwise, you'll need to play with datetime objects ( oh joy ). See next section.
# 
# **Datetime Objects - skip if you're just using Time_Seconds**
# 
# - Datetime objects are specially-formatted to contain dates and times. They're typically the integer number of seconds from 1 Jan, 1970. 
# - The datetime module in Python contains a `datetime` class, so if you're using `import datetime as dt`, then you can call `dt.datetime`
# - This class has the method `strptime`, which produces a datetime object from a string of text. You provide the format. For converting something like `"17/03/2012 18:23"`, you have the format string `"%d/%m/%Y %H:%M"`. If you give `strptime` the datetime string and the format string, it can understand that string and produce a datetime object.
# - From that, you can call that datetime object's `strftime` method, which asks for a formatting string like the one above. If you just give it `"%H"`, for example, it will just give you the hour of that datetime. Using these, you can extract a list of days of the week and of hours of the day from each time entry. See http://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior for a table of formatting strings.
# 
# **Histograms**
# 
# - `plt.hist()` takes a list or array, and produces a histogram. 
# - If you also give it another argument ( `plt.hist(mylist, mybins)` ), then you've provided the boundaries between histogram bins, assuming mybins is a list or array. If it's a scalar, then you're telling it how many bins to use, and it calculates the boundaries itself. 
# 
# **Critical Error - Days without Coffee ?!**
# 
# - Use the function `diff()` in Numpy to look at the difference in time between measurements. 
# - If you spot a large number of days without a measurement, then you've found a time when the coffee machine was broken. This causes critical failure in Warwick Complexity. How many such failures were there ? Look into `np.diff(t)` to find out. 
# - Plot it, against time ( remember that `diff()` will provide one less value, as it gives you the differences between subsequent values ), and generate a histogram to check out the distribution of days between measurements.
# 
# **Always Acknowledge your Coworkers**
# 
# - `plt.pie()` takes a list or array, and produces a pie chart. 
# - The second argument you feed it contains the labels of each slice of pie - so `plt.pie(mydata, mylabels)` will put labels next to each slice. Generate a pie chart, showing who contributed to this time series.
# 
# **Computing an Average Sampling Rate**
# 
# - If you plot time on the y axis ( just against indices ), and on the same plot, have a straight line of gradient 1, you'll note that most of our time measurements ( the times at which we took measurements ) exist under this line. That means that the derivative of the time curve is less than one. Its reciprocal is the sampling rate - the number of measurements taken per unit time.
# - Calculate this sampling rate by using `scipy.stats.linregress` to perform a linear regression. Catch the gradient and intercept, and plot it on top of the time plot, so we can visually check the fit. Also catch the $R^2$, standard error, and the p-value of the regression. 
# - Command : `gradient, intercept, r2, pval, stderr = stats.linregress(indices, time)`
# 
# **Regularisation of the Sampling**
# 
# - Irregular sampling is not fun. The series is fairly well-behaved, so we can interpolate over it fairly comfortably.
# - Use `scipy.interpolate.interp1d(x, y)` to create an interpolant. You can then pass an array of times that you want the interpolated values at. Create a variable `coffees_interp` that is the interpolation over the regularly-spaced time vector `t_interp`, and plot them on top of the normal t vs coffees to ensure that the interpolation was correct.
# 
# **Daily Average Over the Week**
# 
# - Using your shiny new regularly-spaced coffee consumption time series, compute the average number of coffees devoured on each day of the week. Plot it with `plt.errorbar` and approximate 95% confidence intervals from the standard error of the mean using `scipy.stats.sem`.

# <codecell>

# Basic numbercrunching
import numpy as np

# Interpolation, FFTs
from scipy import interpolate, stats

# Plotting
import matplotlib.pyplot as plt
import seaborn
figsize(14, 8)

# Datetime manipulation
# Be warned : functions in classes in modules are BADLY named...
import datetime as dt

# Data stuff I don't know how to use to its full potential, but read_csv() is nice
import pandas as pd

# Pickle, for saving variables
import pickle

# <codecell>

# Let's read the data using Pandas, and look at what it contains.
data = pd.read_csv("CoffeeTimeSeries.csv")
data

# <codecell>

# Let's just plot the time-series first. 
# The easiest way to plot against the date variable is to go via
# a datetime object, which can later be formatted as we want it

# strptime creates a datetime object from a string :
times =  [dt.datetime.strptime(i, "%d/%m/%Y %H:%M") for i in data["Timestamp"].values]

# We now have a list of datetime objects, which we can format as we like.
# Let's create a few items : 
# a list of integers for plotting ( t )
# a list of strings of the day of the week ( weekday )
# a list of hours during the day ( dayhours )
# http://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior

t = np.array([int(i.strftime("%s")) for i in times]) / 86400. # in days
weekday = [int(i.strftime("%w")) for i in times]
weekdaylabels = np.unique([i.strftime("%A") for i in times])
dayhours = [int(i.strftime("%H")) for i in times]

# <codecell>

# OK, we've extracted some useful time data. Let's grab the number of coffees and plot stuff.
coffees = data["Coffees"].values

plt.subplot2grid((2, 2), (0, 0), colspan=2)
plt.plot(t, coffees, linewidth=3)
plt.title("Coffee Consumption Time-Series")
plt.xlabel("Time (datetime seconds)")
plt.ylabel("Number of coffees")

plt.subplot(2, 2, 3)
plt.hist(weekday, 7)
plt.title("Consumption Per Day")
plt.xticks(range(7), ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]);
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.hist(dayhours, np.unique(dayhours))
plt.title("Hours of Consumption During the Day")
plt.xlabel("Hour of Day")
plt.ylabel("Frequency")

plt.tight_layout()

# <codecell>

# So, fairly uniform in terms of quantity consumed per day ( Monday blues ? ).
# Hourly is more interesting : it drops during the day, with an after-lunch spike.
# We can also infer a little about what time people arrive at work !

# Let's go back to the time-series itself. Was the machine broken at any point ?
# Consider over ten days a critical emergency. 

plt.subplot(121)
plt.plot(np.diff(t), linewidth=3)
plt.axhline(10, color = seaborn.color_palette("deep", 3)[2], linewidth=3)
plt.title("Days between coffees")
plt.xlabel("Time (days)")
plt.ylabel("Days between coffees")

plt.subplot(122)
plt.hist(np.diff(t), np.arange(np.diff(t).max()))
plt.axvline(10, color = seaborn.color_palette("deep", 3)[2], linewidth=3)
plt.title("Distribution of Number of Days Between Coffees")
plt.xlabel("Days between coffees")
plt.ylabel("Frequency")

plt.tight_layout()

# <codecell>

# Yes, it's been broken ! We note five occasions :
# five continguous measurement of over ten days between coffees.
# The last spike requires a little context. It's got a spike with three, not one, value
# above the 10-mark. This was because data had more or less stopped being collected
# ( you can see a few days between measurements after the last coffee machine breakdown
# and prior to this big spike ).

# So, if, until that point, measurements were *fairly* regular, we can ask who took them.
names = data["Name"].values
contributors = np.unique(names)
contributions = [np.sum(np.array(names) == i) for i in contributors]

plt.figure(figsize=(12,12))
plt.pie(contributions, labels=contributors, colors=seaborn.color_palette("deep", len(contributors)));

# <codecell>

# So mostly me, with great contributions from Mike Irvine and Sergio Morales - thanks guys !

# Let's take a step back and consider the sampling of the time series. We saw earlier how
# the time between measurements varied a great deal. Let's look at raw sampling times first.
plt.plot(t, linewidth=3)
plt.plot(np.arange(t[0], t[-1]), linewidth=3)
plt.title("Sampling Times")
plt.xlabel("Index")
plt.ylabel("Sampling time")
plt.legend(["Measurements taken", "Linear time"], loc=2)

# <codecell>

# In general, the blue line is under the green line, so we sampled more than once a day. 
# Let's calculate the average sampling frequency by finding the gradient of the line of best fit.
# We'll use the long, middle section, between 250 and 650
gradient, intercept, r2, pval, stderr = stats.linregress(range(250, 600), t[250:600])

plt.plot(t, linewidth=3)
plt.plot(np.arange(700) * gradient + intercept, linewidth=3)
plt.title("Approximate Rate of Sampling : %f per day; $R^2$ = %f, Standard Error = %f" % (1/gradient, r2, stderr))
plt.xlabel("Index")
plt.ylabel("Time at measurement (days)")

# <codecell>

# The sampling is not very regular. Let's interpolate to fix that.
# We'll use a standard linear interpolator.

# First, create a list of floats, starting at the first sample, and 
# ending at or just before the last sample, incrementing in one day intervals
regular_time = np.arange(t[0], t[-1])

# Now let's interpolate the series
coffee_interpolator = interpolate.interp1d(t, coffees)
regular_coffees = coffee_interpolator(regular_time)

# Plot both time series against their respective times to confirm
# and only show every tenth interpolated point
plt.plot(t, coffees)
plt.plot(regular_time[::10], regular_coffees[::10], 'o')
plt.title("Interpolation of the Time-Series")
plt.xlabel("Time (days)")
plt.ylabel("Number of coffees devoured")

# <codecell>

# Now that we have a uniformly-sampled time series, we can start to look at regularities during the week.
# Let's compute the average number of coffees for each day of the week. First, let's look at the series itself.
plt.plot(np.diff(regular_coffees))

# <codecell>

# So we see a few issues. Firstly, the constants are due to interpolating over missing data. 
# We could be clever about it, by reinterpolating after inserting a zero-change point just after long breaks,
# but in the interest of time, we'll just skip them for now. Let's start at 210, and finish at 420.
# The big dip is due to people being away. The time series started in October, which puts the dip
# squarely in the summer.

# So, let's compute that average.

# Declare empty arrays
weekly_average = np.zeros(7)
weekly_stderr = np.zeros(7)

# Iterating over each day of the week
for i in range(7) :
    weekly_average[i] = np.mean(np.diff(regular_coffees[210:420])[i::7])
    weekly_stderr[i] = stats.sem(np.diff(regular_coffees[210:420])[i::7])
    
# Plot with approximate 95% CI
plt.errorbar(range(7), weekly_average, yerr=weekly_stderr * 1.96, linewidth=3)

# <codecell>

# If you got this far in an hour, well done.
# You probably work as hard as Warwick Complexity, whose caffeine consumption doesn't drop to zero over the weekends...

