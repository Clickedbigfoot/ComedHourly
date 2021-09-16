#!/usr/bin/python
# This file defines the class for a naive machine learning model
# It calculates the ideal threshold for predicting if a peak is coming
# within the next hour based on the rate of change in the electricity usage
import argparse
from datetime import date, datetime, time, timedelta

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 343
SECONDS_PER_HOUR = 60 * 60
# Scalers trained to normalize unprocessed data
pjmScaler = None
comedScaler = None
# Scalers trained to standardize unprocessed data
pjmStandardizer = None
comedStandardizer = None
# Keep track of all the best parameter results
results = {}


def getParamString(parameters):
    # Creates string for listing model parameters
    result = ""
    for name in list(parameters.keys()):
        if name == "random_state":
            continue
        result = result + name + ":" + str(parameters[name]) + "\n"
    return result[:-1]


def getDatetimeFromTimestamp(timestamp):
    """
    Produces a datetime object matching the timestamp
    @param timestamp: "YYYY.MM.DD.hr.min"
    @return a datetime object
    """
    times = [int(num) for num in timestamp.split(".")]
    return datetime(times[0],
                    times[1],
                    times[2],
                    hour=times[3],
                    minute=times[4])


def isDesiredLater(minus15, minus10, minus5, current, deltas):
    # Determines if the timestamps are properly spaced apart in terms of time
    # @return True if they are spaced properly
    # False otherwise
    if (
        current[0] - minus15[0] != deltas[15]
        or current[0] - minus10[0] != deltas[10]
        or current[0] - minus5[0] != deltas[5]
    ):
        return False
    return True


def getEarlierTime(samples, i, delta):
    """
    Finds the sample that is the required minutes prior
    to sample i.
    @param samples: list of samples
    @param i: index of the current sample
    @param delta: timedelta object of target difference
    @return desired sample from samples if it exists, otherwise None
    """
    current = samples[i]
    target = current[0] - delta
    for j in range(i - 1, -1, -1):
        previous = samples[j]
        if previous[0] < target:
            # Too far back, it doesn't exist
            return None
        if previous[0] == target:
            return previous
    return None


def getRateOfChange(samples, i, deltaMinutes, span, grid):
    """
    Determines the rate of change at the time deltaMinutes before sample i
    @param samples: the list of samples
    @param i: index in samples for the current time
    @param deltaMinutes: minutes prior to sample i at which we want to
    calculate a rate of change
    @param span: duration in minutes for calculating rate of change
    @param grid: "pjm" or "comed"
    @return the rate of change per minute or None if there is no valid rate
    """
    if grid == "pjm":
        grid = 1
    elif grid == "comed":
        grid = 2
    else:
        raise ValueError("Invalid grid argument")
    delta = timedelta(minutes=deltaMinutes)
    current = samples[i]
    target = current[0] - delta
    for j in range(i, -1, -1):
        previous = samples[j]
        if previous[0] < target:
            return None
        if previous[0] == target:
            # Found the sample at which we want a rate of change
            current = previous
            delta = timedelta(minutes=span)
            target = current[0] - delta
            for k in range(j - 1, -1, -1):
                previous = samples[k]
                if previous[0] < target:
                    return None
                if previous[0] == target:
                    return (current[grid] - previous[grid]) / span
            return None
    return None


def getSecondRateOfChange(samples, i, deltaMinutes, span, spanRoc, grid):
    """
    Determines the rate of change in the rate of change at sample
    deltaMinutes before sample i
    @param samples: the list of samples
    @param i: index in samples for the current time
    @param deltaMinutes: minutes prior to sample i that we want to calculate
    the roc of the roc
    @param span: duration in minutes for calculating the roc of roc\
    @param span: span fo getRateOfChange function
    @param grid: "pjm" or "comed"
    @return the rate of change in rate of change per minute
    or None if there is no valid rate
    """
    delta = timedelta(minutes=deltaMinutes)
    current = samples[i]
    target = current[0] - delta
    for j in range(i, -1, -1):
        previous = samples[j]
        if previous[0] < target:
            return None
        if previous[0] == target:
            # Found the sample at which we want a rate of change
            roc1 = getRateOfChange(samples, j, 0, spanRoc, grid)
            if roc1 is None:
                return None
            roc2 = getRateOfChange(samples, j, span, spanRoc, grid)
            if roc2 is None:
                return None
            return (roc1 - roc2) / span
    return None


def isPositive(currentTimestamp, peaks, grid):
    # Determine if this time should be a positive label
    # @param currentTimestamp: datetime object for the sample
    # @param peaks: dictionary of peak times
    # @param grid: "pjm" or "comed"
    # @return 1 if currentTimestamp is <= 1 hour before its peak, 0 otherwise
    if grid == "pjm":
        peakTime = peaks[currentTimestamp.date()][0]
    elif grid == "comed":
        peakTime = peaks[currentTimestamp.date()][1]
    else:
        raise ValueError("Invalid electricity grid specified in arguments")
    peakTime = datetime.combine(currentTimestamp.date(), peakTime)
    if peakTime < currentTimestamp:
        return 0
    delta = peakTime - currentTimestamp
    if delta.total_seconds() > SECONDS_PER_HOUR:
        return 0
    return 1


def getSplitData(features, labels, dataSplit):
    """
    Splits the data into training and testing data
    @param features: ndarray or the features
    @param labels: array-like of labels
    @return trainData, testData, with labels as the last column of both
    """
    xTrain, xTest, yTrain, yTest = train_test_split(
        features, labels, train_size=dataSplit, stratify=labels,
        random_state=SEED
    )
    yTrain = np.reshape(yTrain, (-1, 1))
    yTest = np.reshape(yTest, (-1, 1))
    trainData = np.concatenate((xTrain, yTrain), axis=1)
    testData = np.concatenate((xTest, yTest), axis=1)
    return trainData, testData


def readData(inputPath):
    """
    Reads in the data and returns a list of the samples and
    a dictionary of the peaks for each valid day.
    The samples only have their datetime object and the load values
    """
    inputFile = open(inputPath, "r")
    line = inputFile.readline()  # Skip header
    line = inputFile.readline()
    peaks = {}  # {datetime : (hour of pjm and comed peaks of that day)}
    samples = []
    # We only need year, month, and day
    placeholderTimestamp = date(1970, 1, 1)
    # Track the peak amount and the time it occurs in
    placeholderPeak = (-1, time(1))
    currentTimestamp = placeholderTimestamp
    peakPjm = placeholderPeak
    peakComed = placeholderPeak
    while line != "":
        line = line.strip().split(",")
        timestamp = getDatetimeFromTimestamp(line[0])
        pjmLoad = int(line[1])
        comedLoad = int(line[2])
        samples.append([timestamp, pjmLoad, comedLoad])
        # Record peaks
        if timestamp.date() != currentTimestamp:
            # Save the old day's peak hours if this is a new day
            peaks[currentTimestamp] = (peakPjm[1], peakComed[1])
            currentTimestamp = timestamp.date()
            peakPjm = (pjmLoad, timestamp.time())
            peakComed = (comedLoad, timestamp.time())
            line = inputFile.readline()
            continue
        if peakPjm[0] < pjmLoad:
            # Save new pjm peak
            peakPjm = (pjmLoad, timestamp.time())
        if peakComed[0] < comedLoad:
            # Save new comed peak
            peakComed = (comedLoad, timestamp.time())
        line = inputFile.readline()
    peaks[currentTimestamp] = (peakPjm[1], peakComed[1])
    inputFile.close()
    # Get rid of placeholder peaks or bad data
    del peaks[placeholderTimestamp]
    badPeaks = []
    for peak in list(peaks.keys()):
        isBad = False
        peakPjm = peaks[peak][0]
        peakComed = peaks[peak][1]
        if peakPjm.hour < 6 or peakPjm.hour > 19:
            isBad = True
        if peakComed.hour < 6 or peakComed.hour > 19:
            isBad = True
        if isBad:
            badPeaks.append(peak)
    goodSamples = []
    for sample in samples:
        if sample[0].date() in badPeaks:
            continue
        goodSamples.append(sample)
    for peak in badPeaks:
        del peaks[peak]
    return goodSamples, peaks


def trainScalers(pjmFeatures, comedFeatures):
    global pjmScaler
    global comedScaler
    pjmScaler = MinMaxScaler().fit(pjmFeatures)
    comedScaler = MinMaxScaler().fit(comedFeatures)
    global pjmStandardizer
    global comedStandardizer
    pjmStandardizer = StandardScaler().fit(pjmFeatures)
    comedStandardizer = StandardScaler().fit(comedFeatures)


def getSamples1(samples, peaks):
    """
    We want features to be the capacity loads at the current,
    5 minutes, 10 minutes, and 15 minutes prior times.
    Labels are whether or not a peak occurs within sixty minutes.
    We need two separate arrays of training data as described above,
    one for predicting PJM peaks and one for predicting Comed peaks.
    @return pjmFeatures, pjmLabels, comedFeatures, comedLabels
    """
    i = 0
    pjm = []
    pjmLabels = []
    comed = []
    comedLabels = []
    deltas = {
        5: timedelta(minutes=5),
        10: timedelta(minutes=10),
        15: timedelta(minutes=15),
        30: timedelta(minutes=30),
    }
    while i < len(samples) - 3:
        # Set i to a proper value first
        current = samples[i + 3]
        minus5 = samples[i + 2]
        minus10 = samples[i + 1]
        minus15 = samples[i]
        if not isDesiredLater(minus15, minus10, minus5, current, deltas):
            i += 1
            continue
        # Create samples with their label
        pjmSample = [current[1], minus5[1], minus10[1], minus15[1]]
        pjmLabels.append([isPositive(current[0], peaks, "pjm")])
        comedSample = [current[2], minus5[2], minus10[2], minus15[2]]
        comedLabels.append([isPositive(current[0], peaks, "comed")])
        pjm.append(pjmSample)
        comed.append(comedSample)
        i += 1
    pjm = np.array(pjm)
    comed = np.array(comed)
    return pjm, pjmLabels, comed, comedLabels


def getFeatures2(samples, current, i, grid):
    """
    Returns a list of the features for sample i
    Return None if there is no valid set of features for this sample
    """
    # Add the raw load at the moment
    roc0 = current[1] if grid == "pjm" else current[2]
    # Add rate of change over last 60 minutes
    roc1 = getRateOfChange(samples, i, 0, 60, grid)
    if roc1 is None:
        return None
    # Add rate of change over last 60 minutes, 30 minutes ago
    roc2 = getRateOfChange(samples, i, 30, 60, grid)
    if roc2 is None:
        return None
    # Add rate of change over last 60 minutes, 60 minutes ago
    roc3 = getRateOfChange(samples, i, 60, 60, grid)
    if roc3 is None:
        return None
    # Add a rate of change in rate of change over the last 60 minutes
    roc4 = getSecondRateOfChange(samples, i, 0, 60, 60, grid)
    if roc4 is None:
        return None
    # Add a rate of change in rate of change over the last 60 minutes,
    # 10 minutes ago
    roc5 = getSecondRateOfChange(samples, i, 10, 60, 60, grid)
    if roc5 is None:
        return None
    return [roc0, roc1, roc2, roc3, roc4]


def getSamples2(samples, peaks):
    """
    We want features to be the capacity loads at the current,
    5 minutes, 10 minutes, and 15 minutes prior times.
    Labels are whether or not a peak occurs within sixty minutes.
    We need two separate arrays of training data as described above,
    one for predicting PJM peaks and one for predicting Comed peaks.
    @return pjmFeatures, pjmLabels, comedFeatures, comedLabels
    @TODO revise this comment block
    """
    i = 2
    pjm = []
    pjmLabels = []
    comed = []
    comedLabels = []
    while i < len(samples):
        # Set i to a proper value first
        current = samples[i]
        pjmFeatures = getFeatures2(samples, current, i, "pjm")
        comedFeatures = getFeatures2(samples, current, i, "comed")
        if pjmFeatures is None or comedFeatures is None:
            i += 1
            continue
        pjm.append(pjmFeatures)
        comed.append(comedFeatures)
        # Create samples with their label
        pjmLabels.append(isPositive(current[0], peaks, "pjm"))
        comedLabels.append(isPositive(current[0], peaks, "comed"))
        i += 1
    pjm = np.array(pjm)
    comed = np.array(comed)
    return pjm, pjmLabels, comed, comedLabels


def preprocess(inputPath, dataSplit, mode):
    # First create a list of each entry in the csv
    # while also collecting the peak hours for each day
    samples, peaks = readData(inputPath)
    print("Initial samples: " + str(len(samples)))
    # Now create training data from the peaks and list of samples
    if mode == 1:
        pjm, pjmLabels, comed, comedLabels = getSamples1(samples, peaks)
    else:
        pjm, pjmLabels, comed, comedLabels = getSamples2(samples, peaks)

    # Split data into training and testing data
    pjmTrain, pjmTest = getSplitData(pjm, pjmLabels, dataSplit)
    comedTrain, comedTest = getSplitData(comed, comedLabels, dataSplit)
    print("Training Samples: " + str(len(pjmTrain)))
    print("Testing Samples: " + str(len(pjmTest)))
    pjmPositive = 0
    comedPositive = 0
    for i in range(0, len(pjmTest)):
        pjmPositive += pjmTest[i][-1]
        comedPositive += comedTest[i][-1]
    print("PJM positive tests: " + str(pjmPositive))
    print("Comed positive tests: " + str(comedPositive))

    # Preprocess features
    trainScalers(pjmTrain[:, :-1], comedTrain[:, :-1])

    return pjmTrain, pjmTest, comedTrain, comedTest


def getBestParameters(estimator, parameters, scoring, trainingData):
    if estimator == "dtree":
        search = GridSearchCV(
            DecisionTreeClassifier(), parameters, scoring=scoring,
            error_score="raise"
        )
    elif estimator == "svc":
        search = GridSearchCV(SVC(), parameters, scoring=scoring,
                              error_score="raise")
    else:
        raise ValueError("Invalid estimator passed to function")
    search = search.fit(trainingData[:, :-1], trainingData[:, -1])
    return search.best_params_


def getModelScore(estimator, parameters, scoring, trainingData, testingData):
    if estimator == "dtree":
        model = DecisionTreeClassifier(
            max_depth=parameters["max_depth"],
            min_samples_split=parameters["min_samples_split"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            random_state=SEED,
            min_impurity_decrease=parameters["min_impurity_decrease"],
        )
    elif estimator == "svc":
        model = SVC(
            C=parameters["C"],
            kernel=parameters["kernel"],
            gamma=parameters["gamma"],
            random_state=SEED,
        )
    else:
        raise ValueError("Invalid estimator passed to function")
    model = model.fit(trainingData[:, :-1], trainingData[:, -1])
    labels = model.predict(testingData[:, :-1])
    recall = round(recall_score(testingData[:, -1], labels), 3)
    precision = round(precision_score(testingData[:, -1], labels), 3)
    return [recall, precision]


def scaleData(scaler, trainData, testData):
    trainDataX = scaler.transform(trainData[:, :-1])
    trainDataY = np.reshape(trainData[:, -1], (-1, 1))
    trainData = np.concatenate((trainDataX, trainDataY), axis=1)
    testDataX = scaler.transform(testData[:, :-1])
    testDataY = np.reshape(testData[:, -1], (-1, 1))
    testData = np.concatenate((testDataX, testDataY), axis=1)
    return trainData, testData


def runDtreeTests(pjmTrain, pjmTest, comedTrain, comedTest):
    # A max_depth too large will overfit,
    # but a max_depth too small will underfit
    maxDepth = [5, 15, 20, 25]
    # Ideal min_split is between 1-40 for sklearn CART implementations of
    # decision trees,
    # so we look in that range. min_split controls overfitting,
    # but too high will cause it to underfit
    minSplit = [2, 5, 10, 20, 30, 40]
    # Ideal min_leaf is between 1-20 and similarly controls
    # overfitting by restricting the tree from making super specific
    # branches for a single sample (or more)
    minLeaf = [1, 5, 10, 15, 20]
    # max_features limits the number of features considered in every split.
    # Helps control overfitting and reduces computation time,
    # But we only have four features to consider at all
    maxFeatures = [None, "log2"]
    # minImpurity generally should be largest when you have a lot of
    # evenly split data, and 0 when all data belongs to the same class
    # We'll keep it small because most data have the same label, 0
    minImpurity = [0.0, 0.01]
    dtreeParameters = {
        "max_depth": maxDepth,
        "min_samples_split": minSplit,
        "min_samples_leaf": minLeaf,
        "max_features": maxFeatures,
        "random_state": [SEED],
        "min_impurity_decrease": minImpurity,
    }

    # Test for prioritizing recall
    pjmBestParam = getBestParameters("dtree", dtreeParameters,
                                     "recall", pjmTrain)
    pjmScore = getModelScore("dtree", pjmBestParam,
                             "recall", pjmTrain, pjmTest)
    comedBestParam = getBestParameters("dtree", dtreeParameters,
                                       "recall", comedTrain)
    comedScore = getModelScore("dtree", comedBestParam,
                               "recall", comedTrain, comedTest)
    results["dtree"] = [[pjmBestParam, pjmScore],
                        [comedBestParam, comedScore]]


def runSvcTests(pjmTrain, pjmTest, comedTrain, comedTest, standardize):
    # Gamma determines how closely it should fit the data
    # with a nonlinear kernel
    gamma = [0.1, 1, 10, 40, 70, 90, 110]
    # C is the penalty parameter. It controls the tradeoff between
    # smooth boundaries and fitting
    c = [0.1, 1, 10, 40, 70, 90, 110]

    # Further preprocess the data
    if standardize:
        pjmTrain, pjmTest = scaleData(pjmStandardizer, pjmTrain, pjmTest)
        comedTrain, comedTest = scaleData(comedStandardizer,
                                          comedTrain, comedTest)
    else:
        pjmTrain, pjmTest = scaleData(pjmScaler, pjmTrain, pjmTest)
        comedTrain, comedTest = scaleData(comedScaler, comedTrain, comedTest)

    # Test rbf kernel
    svcParameters = {"kernel": ["rbf"], "C": c, "gamma": gamma,
                     "random_state": [SEED]}
    pjmBestParam = getBestParameters("svc", svcParameters, "recall", pjmTrain)
    pjmScore = getModelScore("svc", pjmBestParam, "recall", pjmTrain, pjmTest)
    comedBestParam = getBestParameters("svc", svcParameters,
                                       "recall", comedTrain)
    comedScore = getModelScore("svc", comedBestParam,
                               "recall", comedTrain, comedTest)
    results["svc"] = [[pjmBestParam, pjmScore], [comedBestParam, comedScore]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and compare the performances \
                    of various machine learning models."
    )
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        default="./usageData.csv",
        help="Path to the data to use.",
    )
    parser.add_argument(
        "-t",
        dest="dataSplit",
        type=float,
        default=0.9,
        help="Percentage of data to be used as test data. Default=0.9",
    )
    parser.add_argument(
        "-m",
        dest="mode",
        type=int,
        default=2,
        help="Mode for choosing features. Default=2",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        dest="visual",
        help="Visualize the results with matplotlib.",
    )
    parser.add_argument(
        "-d", action="store_true", dest="dtree", help="Use a decision tree."
    )
    parser.add_argument(
        "-s", action="store_true", dest="svc", help="Use an SVC.")
    parser.add_argument(
        "-n",
        action="store_true",
        dest="standardize",
        help="Standardize the data for SVC instead of normalize.",
    )
    args = parser.parse_args()
    if args.mode != 1 and args.mode != 2:
        raise ValueError("Invalid choice of mode for choosing features")
    pjmTrain, pjmTest, comedTrain, comedTest = preprocess(
        args.input, args.dataSplit, args.mode
    )
    if args.dtree:
        runDtreeTests(pjmTrain, pjmTest, comedTrain, comedTest)
    if args.svc:
        runSvcTests(pjmTrain, pjmTest, comedTrain, comedTest, args.standardize)

    # Print and plot reults
    if args.dtree:
        pjmBest = results["dtree"][0]
        comedBest = results["dtree"][1]
        print("=" * 40)
        print("PJM Decision Tree Parameters: " + str(pjmBest[0]))
        print("Recall: " + str(pjmBest[1][0]) +
              "\tPrecision: " + str(pjmBest[1][1]))
        print("Comed Decision Tree Parameters: " + str(comedBest[0]))
        print(
            "Recall: " + str(comedBest[1][0]) +
            "\tPrecision: " + str(comedBest[1][1])
        )
        if args.visual:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
            pjmScores = [
                "Recall\n" + str(pjmBest[1][0]),
                "Precision\n" + str(pjmBest[1][1]),
            ]
            ax[0].title.set_text("PJM Decision Tree")
            ax[0].set_ylim([0, 1.0])
            ax[0].bar(pjmScores, pjmBest[1])
            ax[0].set_xlabel(getParamString(pjmBest[0]))
            comedScores = [
                "Recall\n" + str(comedBest[1][0]),
                "Precision\n" + str(comedBest[1][1]),
            ]
            ax[1].title.set_text("Comed Decision Tree")
            ax[1].set_ylim([0, 1.0])
            ax[1].bar(comedScores, comedBest[1])
            ax[1].set_xlabel(getParamString(comedBest[0]))
            plt.tight_layout(pad=1.0)
            plt.savefig("dtree.pdf")
            plt.show()
    if args.svc:
        pjmBest = results["svc"][0]
        comedBest = results["svc"][1]
        print("=" * 40)
        print("PJM SVC Parameters: " + str(pjmBest[0]))
        print("Recall: " + str(pjmBest[1][0]) +
              "\tPrecision: " + str(pjmBest[1][1]))
        print("Comed SVC Parameters: " + str(comedBest[0]))
        print(
            "Recall: " + str(comedBest[1][0]) +
            "\tPrecision: " + str(comedBest[1][1])
        )
        if args.visual:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
            if args.standardize:
                fig.suptitle("Standardized Data")
            else:
                fig.suptitle("Normalized Data")
            pjmScores = [
                "Recall\n" + str(pjmBest[1][0]),
                "Precision\n" + str(pjmBest[1][1]),
            ]
            ax[0].title.set_text("PJM SVC")
            ax[0].set_ylim([0, 1.0])
            ax[0].bar(pjmScores, pjmBest[1])
            ax[0].set_xlabel(getParamString(pjmBest[0]))
            comedScores = [
                "Recall\n" + str(comedBest[1][0]),
                "Precision\n" + str(comedBest[1][1]),
            ]
            ax[1].title.set_text("Comed SVC")
            ax[1].set_ylim([0, 1.0])
            ax[1].bar(comedScores, comedBest[1])
            ax[1].set_xlabel(getParamString(comedBest[0]))
            plt.tight_layout(pad=1.0)
            plt.savefig("svc.pdf")
            plt.show()
