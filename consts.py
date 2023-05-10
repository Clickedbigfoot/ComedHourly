"""
Constants for various files
"""

########################################
# Define areas to grab weather data for.
# These should be tuples with a name for the db table mapping to a latlon
# tuple of strings.
# Area names should be identical to what is in the database
########################################

AREAS = [('lombard', ('41.871281', '-88.035065')),
         ('nyc', ('40.712778', '-74.006111'))]

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
