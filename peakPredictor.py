#!/usr/bin/python3
"""
Class for predictor object that determines whether or not to turn off
electricity.
Script contains loop for mkaing predictions and scraping data into an SQL database
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import os
import pickle
import signal
import sys
from threading import Event
from time import sleep

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sqlalchemy import create_engine, func
from sqlalchemy import Column, Date, Integer, MetaData, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.sql.expression import and_
from sqlalchemy.types import DateTime

from compareModels import getFeatures, getSamples, readData


# TARGET_URL = "https://datasnapshot.pjm.com/content/InstantaneousLoad.aspx"
# PJM_INDICATOR = "<td>PJM RTO Total</td>\r\n\t\t        <td class=\"right\">"
# PJM_INDICATOR_LEN = len(PJM_INDICATOR)
# COMED_INDICATOR = "<td>COMED Zone</td>\r\n\t\t        <td class=\"right\">"
# COMED_INDICATOR_LEN = len(COMED_INDICATOR)
SECONDS_PER_5_MIN = 300
DATETIME_FORMAT = r"%Y.%m.%d.%H.%M"
START_MONTH = 5 # Start in May
END_MONTH = 10 # Stops predicting in October
# Ratio of current load to 5th highest peak to be a contender
RATIO = 0.1 # @TODO set to 0.8 or something more viable when done testing
PREDICTOR_SAVE = "predictor.pkl"

finish = Event()

# Declare structure for SQLite database

Base = declarative_base()

class Loads(Base):
    __tablename__ = "loads"
    sql_datetime = Column(Integer, primary_key=True)
    pjm = Column(Integer)
    comed = Column(Integer)


class Predictor():
    def __init__(self,
                 database,
                 existingModel=None,
                 trainingData="./usageData.csv"):
        self.engine = None
        self.Session = None
        self.db = self.loadDatabase(database)
        self.pjmModel = None
        self.pjmScaler = None
        self.comedModel = None
        self.comedScaler = None
        if existingModel is None:
            self.trainModel(trainingData)
        else:
            self.loadModel(existingModel)
        self.isFlagged = False
        # Smallest of the top 5 peaks of each grid this year
        self.pjmMinimum = self.getPeakMinimum("pjm")
        self.comedMinimum = self.getPeakMinimum("comed")

    def loadDatabase(self, sqlUri):
        # Sets up engine and sessionmaker for sqlite
        self.engine = create_engine(sqlUri)
        self.Session = sessionmaker()
        self.Session.configure(bind=self.engine)
        with self.Session() as session:
            Base.metadata.create_all(self.engine)

    def update(self):
        # Updates values of the minimum peak values
        self.pjmMinimum = self.getPeakMinimum("pjm")
        self.comedMinimum = self.getPeakMinimum("comed")

    def trainModel(self, trainingData):
        """
        Uses the training data to create all of the predictive and
        scaling models
        """
        # Get training data and scalers
        samples, peaks = readData(trainingData)
        pjmData, pjmLabels, comedData, comedLabels = getSamples(samples, peaks)
        self.pjmScaler = StandardScaler().fit(pjmData)
        self.comedScaler = StandardScaler().fit(comedData)
        pjmData = self.pjmScaler.transform(pjmData)
        comedData = self.comedScaler.transform(comedData)

        # Train classifiers
        self.pjmModel = SVR(C=30, kernel="rbf", gamma=1.5).fit(pjmData,
                                                         pjmLabels)
        self.comedModel = SVR(C=30, kernel="rbf", gamma=1.5).fit(comedData,
                                                           comedLabels)

    def loadModel(self, modelPath):
        # Loads trained models from a saved predictor object
        with open(modelPath, "rb") as inputFile:
            oldModel = pickle.load(inputFile)
        self.pjmModel = deepcopy(oldModel.pjmModel)
        self.pjmScaler = deepcopy(oldModel.pjmScaler)
        self.comedModel = deepcopy(oldModel.comedModel)
        self.comedScaler = deepcopy(oldModel.comedScaler)

    def getPeakMinimum(self, grid):
        # Grabs the fifth largest load for the respecitve grid this summer
        if grid != "pjm" and grid != "comed":
            raise ValueError(f"Invalid grid value '{grid}'")
        latest = datetime.now()
        latest = latest - timedelta(hours=latest.hour, minutes=latest.minute)
        earliest = datetime(latest.year, START_MONTH, 1)
        latest = latest.timestamp()
        earliest = earliest.timestamp()
        with self.Session() as session:
            if grid == "pjm":
                samplesDb = (
                            session.query(Loads)
                            .filter(and_(Loads.sql_datetime >= earliest, Loads.sql_datetime <= latest))
                            .order_by(Loads.pjm.desc())
                            .all())
            else:
                samplesDb = (
                        session.query(Loads)
                        .filter(and_(Loads.sql_datetime >= earliest, Loads.sql_datetime <= latest))
                        .order_by(Loads.comed.desc())
                        .all())
        if len(samplesDb) < 5:
            return None
        fifthPeak = samplesDb[4]
        if grid == "pjm":
            return fifthPeak.pjm
        return fifthPeak.comed

    def turnOffElectricity(self):
        # Return True if a peak occurs in the present hour,
        # false otherwise
        # Determine if this is a summer month
        present = datetime.now()
        earliest = datetime(present.year, START_MONTH, 1)
        latest = datetime(present.year, END_MONTH, 1)
        if not (earliest < present and present < latest):
            print("Invalid month") #@TODO remove
            return self.setFlag(False)
        # See if we have 5 peaks yet
        if self.pjmMinimum is None or self.comedMinimum is None:
            print("Not enough peaks") #@TODO remove
            return self.setFlag(False)

        with self.Session() as session:
            samples = self.getLastSixHours(session, present)
        if len(samples) < 1:
            # Could not retrieve last six hours
            print("Issue: Could not retrieve last six hours")
            print(f"\tat {present}")
            return self.setFlag(False)
        current = samples[-1]

        if current[0].minute == 0:
            # We just turned the hour, so recalculate if a peak is imminent
            if current[1] >= RATIO * self.pjmMinimum:
                pred = self.getPrediction(samples, "pjm")
                if pred <= 55:
                    print(f"PJM peak at new hour {pred}") #@TODO remove
                    return self.setFlag(True)
            if current[2] >= RATIO * self.comedMinimum:
                pred = self.getPrediction(samples, "comed")
                if pred <= 55:
                    print(f"Comed peak at new hour {pred}") #@TODO remove
                    return self.setFlag(True)
            # No peak is imminent
            print("No peak found at new hour") #@TODO remove
            return self.setFlag(False)

        if self.isFlagged:
            print("Already flagged") #@TODO remove
            return self.setFlag(True)

        # Not currently flagged as peak hour, check to decide
        minLeftInHour = 60 - datetime.now().minute
        if current[1] >= RATIO * self.pjmMinimum:
            pred = self.getPrediction(samples, "pjm")
            if pred >= 0 and (pred <= minLeftInHour or pred < 15):
                print(f"Pjm peak {pred}") #@TODO remove
                return self.setFlag(True)
        if current[2] >= RATIO * self.comedMinimum:
            pred = self.getPrediction(samples, "comed")
            if pred >= 0 and (pred <= minLeftInHour or pred < 15):
                print(f"Comed peak {pred}") #@TODO remove
                return self.setFlag(True)
        print("No peak found in middle of hour") #@TODO remove
        return self.setFlag(False)

    def setFlag(self, decision):
        self.isFlagged = decision
        return decision

    def getPrediction(self, samples, grid):
        # Makes a prediction using the sklearn models
        if grid != "pjm" and grid != "comed":
            raise ValueError(f"Invalid grid '{grid}")
        i = len(samples) - 1
        features = getFeatures(samples, samples[-1], i, grid)
        if features is None:
            print("[")
            for sample in samples:
                print("\t" + str(sample))
            print("]" + str(len(samples)))
            print(i)
            print("Issue: Couldn't get features to make prediction")
            print(f"\tat datetime {samples[-1][0]}")
            return False
        features = np.array([features])
        if grid == "pjm":
            features = self.pjmScaler.transform(features)
            pred = self.pjmModel.predict(features)
        else:
            features = self.comedScaler.transform(features)
            pred = self.comedModel.predict(features)
        if pred.shape != (1,):
            raise AssertionError(f"Wrong prediction shape {pred.shape}")
        return pred[0]

    def getLastSixHours(self, session, dt):
        # Gets a list of samples from the last six hours
        # @param dt: the current datetime
        earliest = getTimestamp(dt - timedelta(hours=6, minutes=5))
        samplesDb = (
                        session.query(Loads)
                        .filter(and_(Loads.sql_datetime >= earliest, Loads.sql_datetime <= getTimestamp(dt)))
                        .order_by(Loads.sql_datetime.asc())
                        .all())
        samples = []
        for sample in samplesDb:
            timestamp = datetime.fromtimestamp(sample.sql_datetime)
            pjmLoad = sample.pjm
            comedLoad = sample.comed
            samples.append([timestamp, pjmLoad, comedLoad])
        return samples


def signalHandler(sig, frame):
    print("SIGINT received. Exiting soon")
    finish.set()


def getSecondsLeft():
    present = datetime.now()
    seconds = -(present.minute * 60 + present.second) % SECONDS_PER_5_MIN
    delta = timedelta(seconds=seconds)
    # Return a little more seconds than necessary just to avoid
    # math issues
    target = present + delta - timedelta(microseconds=present.microsecond)
    if target.minute % 5 != 0:
        raise AssertionError("Timestamp not multiple of 5")
    # Add 6 seconds to give CheckUsage.exe time to update the csv
    # If we upgrade this script to look up the load data itself, we
    # can reduce this to 2 seconds instead. It's not a big deal though
    return target, seconds + 30


def getLastEntry(inputPath="./usageData.csv"):
    lastEntry = ""
    with open(inputPath, "rb") as inputFile:
        inputFile.seek(-2, os.SEEK_END)
        while inputFile.read(1) != b'\n':
            inputFile.seek(-2, os.SEEK_CUR)
        lastEntry = inputFile.readline().decode()
    return lastEntry


def getData(nextEntryTime, Session, csvFile="./usageData.csv"):
    """
    Adds the latest addition to the csv file to the SQL database
    Decides whether or not to make a prediction with the predictor
    @param nextEntryTime: datetime of the target entry
    @param Session: sessionmaker for SQLite
    @param csvFile: csv file from which to read
    @return 0 if everything was without issue, -1 otherwise
    """
    # Get current sample
    lastEntry = getLastEntry(csvFile).strip()
    if len(lastEntry) < 1:
        print("Issue: Couldn't read last entry in csv")
        return -1
    lastEntry = lastEntry.split(",")
    times = [int(num) for num in lastEntry[0].split(".")]
    dt = datetime(times[0], times[1], times[2], hour=times[3], minute=times[4])
    # Check if this matches the intended datetime
    if nextEntryTime != dt:
        print("\t" + str(nextEntryTime))
        print("\t" + str(dt))
        print("\t" + str(lastEntry))
        print("Issue: Incorrect last entry in csv")
        return -1
    # Check if time already exists in database
    timestamp = getTimestamp(dt)
    with Session() as session:
        sampleDb = (
                    session.query(Loads)
                    .filter(Loads.sql_datetime == timestamp)
                    .one_or_none()
                    )
        if sampleDb is not None:
            print("Issue: Sample already loaded into SQL database")
            return -1
        # Add to database
        pjmLoad = int(lastEntry[1])
        comedLoad = int(lastEntry[2])
        newLoad = Loads(sql_datetime=timestamp, pjm=pjmLoad, comed=comedLoad)
        session.add(newLoad)
        session.commit()
    return 0


def getTimestamp(dt):
    dt = dt - timedelta(seconds=dt.second, microseconds=dt.microsecond)
    return dt.timestamp()


def populateDatabase(Session, csvFile):
    samples = []
    inputFile = open(csvFile, "r")
    line = inputFile.readline()
    line = inputFile.readline()
    while line != "":
        line = line.strip().split(",")
        times = [int(num) for num in line[0].split(".")]
        samples.append([datetime(times[0], times[1], times[2], hour=times[3], minute=times[4]), int(line[1]), int(line[2])])
        line = inputFile.readline()
    inputFile.close()
    with Session() as session:
        for sample in samples:
            timestamp = sample[0]
            pjmLoad = sample[1]
            comedLoad = sample[2]
            # Check if entry exists already
            timestampDb = (
                           session.query(Loads)
                           .filter(Loads.sql_datetime == getTimestamp(timestamp))
                           .one_or_none()
                           )
            if timestampDb is not None:
                continue
            # Add this timestamp to it
            newLoad = Loads(sql_datetime=getTimestamp(timestamp), pjm=pjmLoad, comed=comedLoad)
            session.add(newLoad)
        session.commit()

def handleDecision(decision):
    # Responds to the decision of the predictor
    # @TODO Have this interface with a usb device to output a signal
    outputFile = open("log.csv", "a+")
    log = f"{datetime.now().strftime(DATETIME_FORMAT)},"
    log = log + str(int(decision)) + "\r\n"
    outputFile.write(log)
    outputFile.close()

def main(args):
    signal.signal(signal.SIGINT, signalHandler)
    cwd = os.getcwd()
    dbPath = os.path.join(cwd, args.database)
    print(f"Using database at {dbPath}")
    sqlUri = f"sqlite:///{dbPath}"
    engine = create_engine(sqlUri)
    Session = sessionmaker()
    Session.configure(bind=engine)

    if args.populate != "":
        Base.metadata.create_all(engine)
        populateDatabase(Session, args.populate)
        print("SQL database populated")
        exit()

    print("Training predictor")
    predictor = Predictor(sqlUri, trainingData=args.training)
    print("Finished training")

    print("Press CTRL+C to shut down the program")
    nextEntryTime, seconds = getSecondsLeft()
    finish.wait(seconds)
    while not finish.is_set():
        if getData(nextEntryTime, Session, csvFile=args.csv) == 0:
            handleDecision(predictor.turnOffElectricity())
        nextEntryTime, seconds = getSecondsLeft()
        finish.wait(seconds)

    # Save predictor using pickle and remove unpickleable instances
    predictor.engine = None
    predictor.Session = None
    with open(PREDICTOR_SAVE, "wb") as outputFile:
        pickle.dump(predictor, outputFile, pickle.HIGHEST_PROTOCOL)
    print("Fini")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict when a peak is emminent."
    )
    parser.add_argument(
        "-t",
        dest="training",
        type=str,
        default="./usageData.csv",
        help="Path to the training data to use.",
    )
    parser.add_argument(
        "-c",
        dest="csv",
        type=str,
        default="usageData.csv",
        help="Path to csv file to read latest sample from. usageData.csv "
             "by default",
    )
    parser.add_argument(
        "-d",
        dest="database",
        type=str,
        default="storedLoads.db",
        help="Path to the SQLite database in this folder",
    )
    parser.add_argument(
        "-p",
        dest="populate",
        type=str,
        default="",
        help="Path to csv file to preload into database. None by default",
    )
    args = parser.parse_args()
    main(args)