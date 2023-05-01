"""
Purpose: Comunicate with weather API to retrieve relevant information.

This will use the weather API at https://www.weatherapi.com

In the API's portal, you can specify all of the information you
would like returned in the response to calls. In order for this
class to work, the appropriate details must be selected.

Note: if no location is ever provided, this 
"""
import datetime
import requests


# Base URL for every API request
BASE_URL = "http://api.weatherapi.com/v1"

class RequestError(Exception):
    """Raise when an HTTP error is returned."""

    def __init__(self, http_code, api_code=None, msg=None):
        """
        Initialize error.

        @param http_code: int representing http error code
        @param api_code: string of api error code if any is returned
        @param msg: error message returned by API if any
        """
        self.http_code = http_code
        self.api_code = api_code
        self.message = msg if msg else 'http error {}'.format(http_code)
        super().__init__(self.message)


class UnexpectedResponseError(Exception):
    """Raise when expected format/info is not found in a response."""

    def __init__(self,
                 msg='expected format/info not found in response',
                 text=''):
        """
        Initialize error.

        @param msg: message explaining exception
        @param text: string raw text of the response's payload
        """
        self.message = msg
        self.text = text
        super().__init__(self.message)


class StaleDataError(Exception):
    """Raise when get_current_weather() gets info that is too old."""

    def __init__(self, tolerance, difference, current):
        """
        Initialize error.

        @param tolerance: int maximum number of minutes old that the
            data can be.
        @param timedelta object representing current time - data's time
        @param current: dictionary representing Current json object from
            the api response
        """
        self.tolerance = tolerance
        self.difference = difference
        self.current = current
        n_minutes = (difference.seconds // 60) + difference.days * 1440
        self.message = ('tolerance of {} minutes exceeded by {} '
                        'minutes'.format(tolerance, n_minutes - tolerance))
        super().__init__(self.message)


class WeatherApi():
    """
    Grab weather information from API.
    """

    def __init__(self, api_key, zip_code=None, latlon=None,
                 temp_unit='farenheit', tolerance=60):
        """
        Initialize instance variables.

        This constructor initializes the API key that this object needs
        to make its requests. It can also have a default location for
        all of its requests, but each method will have optional arguments
        to overrule what is hear. If no location is provided here and no
        arguments are provided to the method calls, then the location will
        automatically use the IP address.

        @param api_key: API key
        @param zip_code: string zip code for target area. Overruled by
            latlon
        @param latlon: tuple of strings for lat and lon, ('lat', 'lon').
            Overrules zip_code.
        @param temp_unit: either celcius or farenheit for temperatures
            returned. Defaults to farenheit
        """
        self.API_KEY = api_key
        self.zip_code = zip_code
        self.latlon = latlon

        temp_unit = temp_unit.lower()
        if temp_unit == 'farenheit':
            self.temp_unit = 'temp_f'
            self.feels_unit = 'feelslike_f'
        elif temp_unit == 'celcius':
            self.temp_unit = 'temp_c'
            self.feels_unit = 'feelslike_c'
        else:
            raise ValueError('invalid unit of temperature')

    def get_current_weather(self, info=None, zip_code=None, latlon=None,
                            tolerance=60):
        """
        Get the current weather.

        @param info: list of strings specifying any of the following
            options as information to retrieve:
            ['temperature', 'feelslike', 'humidity', 'wind_kph']
            By default grabs only the temperature
        @param zip_code: string zip code of area of interest. Overrules
            any default locations defined in the constructor.
        @param latlon: tuple of strings for lat and lon, ('lat', 'lon').
            Overrules zip_code and any default locations defined in the
            constructor.
        @param tolerance: number of minutes old that weather information
            returned is allowed to be.
        @return tuple consisting of
            1) dictionary of the weather information,
            2) dictionary of location information
        """
        # Determine target information
        targets = []
        if info is None:
            info = ['temperature']
        for target in info:
            if target == 'temperature':
                targets.append(self.temp_unit)
            elif target == 'feelslike':
                targets.append(self.feels_unit)
            elif target in {'humidity', 'wind_kph'}:
                targets.append(target)
            else:
                raise ValueError('invalid weather query: {}'.format(target))

        # Make request
        url = '{}{}'.format(BASE_URL, '/current.json')
        response = self.request(url, q=self.get_q(zip_code, latlon))

        # Determine tolerance and current time
        tolerance = int(tolerance)  # In case user puts in a float
        tolerance_td = datetime.timedelta(minutes=tolerance)
        now = datetime.datetime.now(datetime.timezone.utc)

        # Get information of interest
        try:
            # Convert json to dictionary
            json = response.json()

            # Check to see if data is too old
            last_updated = datetime.datetime.fromtimestamp(
                int(json['current']['last_updated_epoch']),
                datetime.timezone.utc)
            difference = now - last_updated
            if now - last_updated > tolerance_td:
                raise StaleDataError(tolerance, difference, json['current'])

            # Return temperature and location
            ret_info = {}
            for i in range(0, len(targets)):
                ret_info[info[i]] = json['current'][targets[i]]
            return (ret_info, json['location'])

        except StaleDataError as e:
            # So that this isn't caught and misreported below
            raise e
        except Exception:
            raise UnexpectedResponseError(text=response.text)

    # @TODO finish this function
#    def get_historical_temperature(self, time, zip_code=None, latlon=None):
#        """
#        Get temperature from a specific time.
#
#        @param time: datetime object representing the time of interest or
#            a string representing that time in format 'YYYY-MM-DD HH:MM'
#            in 24hr clock format. Times must be in UTC.
#        @param zip_code: string zip code of area of interest. Overrules
#            any default locations defined in the constructor.
#        @param latlon: tuple of strings for lat and lon, ('lat', 'lon').
#            Overrules zip_code and any default locations defined in the
#            constructor.
#        @return tuple of the temperature as a float and location
#            information as a dict
#        """
#        # Determine range of time to query as unix timestamps
#        if isinstance(time, str):
#            try:
#                time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M')
#            except Exception:
#                raise ValueError('invalid time format: {}'.format(time))
#        elif not isinstance(time, datetime.datetime):
#            # Not a string or datetime object, invalid type
#            raise TypeError('time arg must be string or datetime object')
#        time = time.replace(tzinfo=datetime.timezone.utc)
#        unix_ts = time.timestamp()
#        
#        # Make request
#        url = '{}{}'.format(BASE_URL, '/history.json')
#        response = self.request(url,
#                                q=self.get_q(zip_code, latlon),
#                                unixdt=unix_ts)
#
#        # Return information of interest
#        try:
#            json = response.json()
#            with open("temp.txt", "w+")as outputFile:
#                for element in json['forecast']['forecastday'][0]['hour']:
#                    outputFile.write("{}\n\n".format(element))
#            exit()
#            return (json['current'][self.temp_unit], json['location'])
#        except Exception as e:
#            print(e)
#            raise UnexpectedResponseError(text=response.text)

    def get_q(self, zip_code, latlon):
        """
        Determine q for request.

        Order of priority for location:
        1) latlon parameter
        2) zip_code parameter
        3) Default latlon instance variable
        4) Default zip_code instance variable
        5) 'auto' which uses the IP address for location
        @return string for the q parameter of the API request
        """
        if latlon is not None:
            return '{},{}'.format(latlon[0], latlon[1])
        if zip_code is not None:
            return zip_code
        if self.latlon is not None:
            return '{},{}'.format(self.latlon[0], self.latlon[1])
        if self.zip_code is not None:
            return self.zip_code
        return 'auto:ip'

    def request(self, url, **kwargs):
        """
        Make request to API.

        @param url: the url for the API request
        @param kwargs: any parameters to the API request, excluding
           the API key
        @return response object from request
        """
        # Create payload of parameters including API key
        payload = {'key': self.API_KEY}
        for key in list(kwargs.keys()):
            payload[key] = kwargs[key]

        # Make request
        response = requests.get(url, params=payload)
        if response.status_code != 200:
            # Check 404 first
            if response.status_code == 404:
                raise RequestError(404)

            # Get json dictionary of response
            try:
                json = response.json()
            except Exception as e:
                raise RequestError(response.status_code)

            # Raise exception with API error details if possible
            if 'error' in json and 'message' in json['error']:
                raise RequestError(response.status_code,
                                   api_code=json['error']['code'],
                                   msg=json['error']['message'])
            raise RequestError(response.status_code)
        return response


if __name__ == '__main__':
    from API_KEY import API_KEY
    api_key = API_KEY
    api = WeatherApi(api_key)
    try:
        answer = api.get_current_weather(('temperature', 'humidity'))
    except UnexpectedResponseError as e:
        print('Ran into response format problems')
        #print(e.text)
    else:
        print(answer)
