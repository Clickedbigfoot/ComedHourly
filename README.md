# Comed Hourly

This is a project designed to take advantage of Comed's hourly pricing plan. To do so, this project aims to train machine learning models that will predict when the peak load hour will be on a given day. These models are then integrated into a system that will automate reactions to such predictions (Such as turning off the air conditioning).

This project is largely compatible with Windows, Mac, and Linux, but will be geared towards Linux.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all requirements.

```bash
pip install -r requirements.txt
```

This project also uses [sqlite](https://sqlite.org/index.html) for storing data. It's compatible with Windows and Mac, but can be installed on Linux with the following:

```bash
sudo apt update
sudo apt install sqlite3
```

## Usage
Atm, you mostly don't.
```python
pass
```

## Contributing

Feel free to open issues/PRs, or repurpose parts of this project for yourself in a personal repo.

## TODO

1) Create python class for handling pjm's new, upcoming API.
2) Create script to scrape data.
3) Flesh out weather_api.py for history data and put together script to backfill as much data as possible.
4) Create scripts to train and test ARIMA machine learning model.
5) Create automated system using trained ARIMA model.
6) Buff sql_io.py to generate csv files on demand.
7) Maybe create proper systemd service and debian package
