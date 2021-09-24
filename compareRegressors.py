#!/usr/bin/python
"""
This compares regressor models.
When a classifier predicts that a peak is within 60 minutes, a
regressor will predict when exactly the peak is. This allows
us to focus training data only on data that is within 60 minutes
of a peak, especially because data before then probably isn't that
indicative
"""
import argparse
from datetime import datetime, date, timedelta, time

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

from compareModels import getAverageLoad, getDatetimeFromTimestamp, \
                          getFeatures, getParamString, getRateOfChange, \
                          getSplitData, readData, scaleData
from compareModels import COMED_MIN, PJM_MIN


SEED = 343
METRIC = "neg_root_mean_squared_error"
SECONDS_PER_HOUR = 60 * 60
# Scalers trained to normalize unprocessed data
pjmScaler = None
comedScaler = None
# Scalers trained to standardize unprocessed data
pjmStandardizer = None
comedStandardizer = None
# Keep track of all the best parameter results
results = {}


def getTimeUntilPeak(currentTimestamp, peaks, grid):
    if grid == "pjm":
        peakLoad = peaks[currentTimestamp.date()][2]
        if peakLoad < PJM_MIN:
            return -1
        peakTime = peaks[currentTimestamp.date()][0]
    elif grid == "comed":
        peakLoad = peaks[currentTimestamp.date()][3]
        if peakLoad < COMED_MIN:
            return -1
        peakTime = peaks[currentTimestamp.date()][1]
    else:
        raise ValueError("Invalid electricity grid specified in arguments")
    peakTime = datetime.combine(currentTimestamp.date(), peakTime)
    if peakTime < currentTimestamp:
        # We already passed the peak
        return -1
    delta = peakTime - currentTimestamp
    return round(delta.total_seconds() / 60)


def trainScalers(pjmFeatures, comedFeatures):
    global pjmScaler
    global comedScaler
    pjmScaler = MinMaxScaler().fit(pjmFeatures)
    comedScaler = MinMaxScaler().fit(comedFeatures)
    global pjmStandardizer
    global comedStandardizer
    pjmStandardizer = StandardScaler().fit(pjmFeatures)
    comedStandardizer = StandardScaler().fit(comedFeatures)


def getSamples(samples, peaks):
    """
    We want features to be the current capacity load, the current rate
    of change over 60 minutes, the rate of change 30 minutes ago, and
    the rate of change 1 hour ago
    """
    i = 2
    pjm = []
    pjmLabels = []
    comed = []
    comedLabels = []
    while i < len(samples):
        current = samples[i][0]
        # Add pjm sample
        pjmCountdown = getTimeUntilPeak(current, peaks, "pjm")
        if 0 <= pjmCountdown and pjmCountdown <= 90:
            pjmFeatures = getFeatures(samples, i, "pjm")
            if pjmFeatures is not None:
                pjm.append(pjmFeatures)
                pjmLabels.append(pjmCountdown)
        # Add comed sample
        comedCountdown = getTimeUntilPeak(current, peaks, "comed")
        if 0 <= comedCountdown and comedCountdown <= 90:
            comedFeatures = getFeatures(samples, i, "comed")
            if comedFeatures is not None:
                comed.append(comedFeatures)
                comedLabels.append(comedCountdown)
        i += 1
    pjm = np.array(pjm)
    comed = np.array(comed)
    return pjm, pjmLabels, comed, comedLabels


def preprocess(inputPath, dataSplit, testPath):
    # First create a list of each entry in the csv
    # while also collecting the peak hours for each day
    samples, peaks = readData(inputPath)
    print("Initial samples: " + str(len(samples)))
    # Now create training data from the peaks and list of samples
    pjm, pjmLabels, comed, comedLabels = getSamples(samples, peaks)

    if testPath == "":
        # Split data into training and testing data
        pjmTrain, pjmTest = getSplitData(pjm, pjmLabels, dataSplit)
        comedTrain, comedTest = getSplitData(comed, comedLabels, dataSplit)
    else:
        # Load separate csv file for testing data
        pjmLabels = np.reshape(pjmLabels, (-1, 1))
        pjmTrain = np.concatenate((pjm, pjmLabels), axis=1)
        comedLabels = np.reshape(comedLabels, (-1, 1))
        comedTrain = np.concatenate((comed, comedLabels), axis=1)
        
        samples, peaks = readData(testPath)
        pjm, pjmLabels, comed, comedLabels = getSamples(samples, peaks)
        pjmLabels = np.reshape(pjmLabels, (-1, 1))
        pjmTest = np.concatenate((pjm, pjmLabels), axis=1)
        comedLabels = np.reshape(comedLabels, (-1, 1))
        comedTest = np.concatenate((comed, comedLabels), axis=1)
        
        pjmTrain = shuffle(pjmTrain, random_state=SEED)
        pjmTest = shuffle(pjmTest, random_state=SEED)
        comedTrain = shuffle(comedTrain, random_state=SEED)
        comedTest = shuffle(comedTest, random_state=SEED)
    
    print("PJM Training Samples: " + str(len(pjmTrain)))
    print("PJM Testing Samples: " + str(len(pjmTest)))
    print("Comed Training Samples: " + str(len(comedTrain)))
    print("Comed Testing Samples: " + str(len(comedTest)))

    # Preprocess features
    trainScalers(pjmTrain[:, :-1], comedTrain[:, :-1])

    return pjmTrain, pjmTest, comedTrain, comedTest


def getBestParameters(estimator, parameters, scoring, trainingData):
    if estimator == "dtree":
        search = GridSearchCV(
            DecisionTreeRegressor(), parameters, scoring=scoring,
            error_score="raise"
        )
    elif estimator == "svr":
        search = GridSearchCV(SVR(), parameters, scoring=scoring,
                              error_score="raise")
    else:
        raise ValueError("Invalid estimator passed to function")
    search = search.fit(trainingData[:, :-1], trainingData[:, -1])
    return search.best_params_


def getModelScore(estimator, parameters, trainingData, testingData):
    if estimator == "dtree":
        model = DecisionTreeRegressor(
            max_depth=parameters["max_depth"],
            min_samples_split=parameters["min_samples_split"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            random_state=SEED,
            min_impurity_decrease=parameters["min_impurity_decrease"],
        )
    elif estimator == "svr":
        model = SVR(
            C=parameters["C"],
            kernel=parameters["kernel"],
            gamma=parameters["gamma"],
        )
    else:
        raise ValueError("Invalid estimator passed to function")
    model = model.fit(trainingData[:, :-1], trainingData[:, -1])
    labels = model.predict(testingData[:, :-1])
    rmse = mse(testingData[:,-1], labels, squared=False)
    """print("="*40)
    for i in range(0, len(labels)):
        print(f"Pred:{labels[i]}\tTarget:{testingData[:,-1][i]}")
    print("="*40)"""
    return rmse


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
                                     METRIC, pjmTrain)
    pjmScore = getModelScore("dtree", pjmBestParam,
                             pjmTrain, pjmTest)
    comedBestParam = getBestParameters("dtree", dtreeParameters,
                                       METRIC, comedTrain)
    comedScore = getModelScore("dtree", comedBestParam,
                               comedTrain, comedTest)
    results["dtree"] = [[pjmBestParam, pjmScore],
                        [comedBestParam, comedScore]]


def runSvrTests(pjmTrain, pjmTest, comedTrain, comedTest, standardize):
    # Gamma determines how closely it should fit the data
    # with a nonlinear kernel
    gamma = [0.1, 0.5, 1, 1.5, 2.5, 5]
    # C is the penalty parameter. It controls the tradeoff between
    # smooth boundaries and fitting
    c = [0.1, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #c = c[:-7] #-7 to get rid of all after 30

    # Further preprocess the data
    if standardize:
        pjmTrain, pjmTest = scaleData(pjmStandardizer, pjmTrain, pjmTest)
        comedTrain, comedTest = scaleData(comedStandardizer,
                                          comedTrain, comedTest)
    else:
        pjmTrain, pjmTest = scaleData(pjmScaler, pjmTrain, pjmTest)
        comedTrain, comedTest = scaleData(comedScaler, comedTrain, comedTest)

    # Test rbf kernel
    svrParameters = {"kernel": ["rbf"], "C": c, "gamma": gamma}
    pjmBestParam = getBestParameters("svr", svrParameters, METRIC, pjmTrain)
    pjmScore = getModelScore("svr", pjmBestParam, pjmTrain, pjmTest)
    comedBestParam = getBestParameters("svr", svrParameters,
                                       METRIC, comedTrain)
    comedScore = getModelScore("svr", comedBestParam,
                               comedTrain, comedTest)
    results["svr"] = [[pjmBestParam, pjmScore], [comedBestParam, comedScore]]


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
        "-x",
        dest="dataSplit",
        type=float,
        default=0.9,
        help="Percentage of data to be used as test data. Default=0.9",
    )
    parser.add_argument(
        "-t",
        dest="test",
        type=str,
        default="",
        help="Path to csv file to use as separate testing data. \
              Splits training data by default",
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
        "-s", action="store_true", dest="svr", help="Use an SVR.")
    parser.add_argument(
        "-n",
        action="store_true",
        dest="standardize",
        help="Standardize the data for SVR instead of normalize.",
    )
    args = parser.parse_args()
    pjmTrain, pjmTest, comedTrain, comedTest = preprocess(
        args.input, args.dataSplit, args.test
    )
    if args.dtree:
        runDtreeTests(pjmTrain, pjmTest, comedTrain, comedTest)
    if args.svr:
        runSvrTests(pjmTrain, pjmTest, comedTrain, comedTest, args.standardize)

    # Print and plot reults
    if args.dtree:
        pjmBest = results["dtree"][0]
        comedBest = results["dtree"][1]
        print("=" * 40)
        print("PJM Decision Tree Parameters: " + str(pjmBest[0]))
        print("RMSE: " + str(pjmBest[1]))
        print("Comed Decision Tree Parameters: " + str(comedBest[0]))
        print(
            "RMSE: " + str(comedBest[1])
        )
        if args.visual:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            models = ["PJM\n" + getParamString(pjmBest[0]),
                    "Comed\n" + getParamString(comedBest[0])]
            ax.title.set_text("Decision Tree Regressors")
            ax.set_ylim([0, 10.0])
            ax.bar(models, [pjmBest[1], comedBest[1]])
            ax.set_xlabel("Model Corresponding to Each Grid")
            ax.set_ylabel("Root mean squared error")
            plt.tight_layout(pad=1.0)
            plt.savefig("dtreeR.pdf")
            plt.show()
    if args.svr:
        pjmBest = results["svr"][0]
        comedBest = results["svr"][1]
        print("=" * 40)
        print("PJM SVR Parameters: " + str(pjmBest[0]))
        print("RMSE: " + str(pjmBest[1]))
        print("Comed SVR Parameters: " + str(comedBest[0]))
        print(
            "RMSE: " + str(comedBest[1])
        )
        if args.visual:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            if args.standardize:
                fig.suptitle("Standardized Data")
            else:
                fig.suptitle("Normalized Data")
            models = ["PJM\n" + getParamString(pjmBest[0])
                      + "\n" + str(pjmBest[1]),
                      "Comed\n" + getParamString(comedBest[0])
                      + "\n" + str(comedBest[1])]
            ax.title.set_text("Support Vector Regressors")
            ax.set_ylim([0, 10.0])
            ax.bar(models, [pjmBest[1], comedBest[1]])
            ax.set_xlabel("Model Corresponding to Each Grid")
            ax.set_ylabel("Root mean squared error")
            plt.tight_layout(pad=1.0)
            plt.savefig("svr.jpg")
            plt.show()
