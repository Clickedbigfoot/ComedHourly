"""
Constants for various files
"""

########################################
# Define areas to grab weather data for.
# These should be tuples with a name for the db table mapping to a latlon
# tuple of strings.
# Area names should be identical to what is in the database
########################################

         # Comed cities
AREAS = [('chicago', ('41.881832', '-87.623177')),
         ('rockford', ('42.259445', '-89.064445')),
         # Cities that represent the rest of PJM's coverage
         ('nyc', ('40.712778', '-74.006111')),
         ('dc', ('38.89511', '-77.03637')),
         ('cincinnati', ('39.103119', '-84.512016')),
         ('columbus', ('39.983334', '-82.983330')),
         ('pittsburgh', ('40.440624', '-79.995888')),
         ('cleveland', ('41.505493', '-81.681290')),
         ('richmond', ('37.541290', '-77.434769')),
         ('virginia_beach', ('36.863140', '-76.015778'))]

#######################################
# For each location table, we want these columns.
# The key of this dict should be the column name mapping to a dict of
# the columns details.
# Each column should include the parameter to query the weather api for
# (as it's defined in the WeatherApi class), if it comes from the api,
# and the sqlite datatype
#######################################

LOCATION_TABLE_COLUMNS = {'sample_id': {'datatype': 'integer unique'},
                          'temperature': {'api_name': 'temperature',
                                          'datatype': 'real'},
                          'feelslike': {'api_name': 'feelslike',
                                        'datatype': 'real'},
                          'humidity': {'api_name': 'humidity',
                                       'datatype': 'integer'},
                          'wind_kph': {'api_name': 'wind_kph',
                                       'datatype': 'real'}}

#######################################
# Describe loads table columns and samples table columns below.
#######################################

LOADS_TABLE_COLUMNS = {'sample_id': {'datatype': 'integer unique'},
                       'comed': {'datatype': 'integer'},
                       'pjm': {'datatype': 'integer'}}

SAMPLES_TABLE_COLUMNS = {'unix_timestamp': {'datatype': 'integer unique'}}

# This determines how stale the data can be when collecting current
# weather data
TOLERANCE = 6  # minutes

# This determines extra seconds we will wait before collecting data to
# give the data sources a chance to update in time
GRACE_PERIOD = 150  # 2.5 minutes

# File to which we log stuff
LOG_FILE = '/var/log/comed_hourlyd.log'

# Map load column name to "load area" of pjm API
LOAD_CLMN_TO_LOAD_AREA = {'pjm': 'PJM RTO', 'comed': 'COMED'}

#######################################
# Describe the days of the week and hours that this script operates.
# Days should be the first three letters with the first capitalized.
# Hours should be in the computer's local time, 24hr.
# Ex) 19:46
# Ex) 02:07
#######################################

VALID_DAYS = {'Mon', 'Tue', 'Wed', 'Thu', 'Fri'}
START_TIME = '06:00'
END_TIME = '19:00'

# Path to sqlite database
DATABASE = 'weatherData.db'
