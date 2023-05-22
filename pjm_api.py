"""
Purpose: This class will facilitate API requests to PJM's Data Miner API.

This allows us to get electricity load data for PJM, Comed, etc.

The API can be found here: https://apiportal.pjm.com
                           https://tools.pjm.com/tools
"""
import consts
import datetime
import requests

BASE_URL = ('https://api.pjm.com/api/v1/inst_load?download=false&'
            'rowCount=1&sort=datetime_beginning_utc&order=desc'
            '&startRow=1&area={}')


class RequestError(Exception):
    """Raise when an error occurs during an api request."""

    def __init__(self, http_code=None, msg=None):
        """
        Initialize error.

        @param http_code: int representing http error code if any
        @param msg: string message for exception
        """
        self.http_code = http_code
        if http_code is not None:
            self.message = 'http error {}'.format(http_code)
        elif msg is not None:
            self.message = msg
        else:
            self.message('error when making api request')
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

    def __init__(self, tolerance, last_update, current):
        """
        Initialize error.

        @param tolerance: int maximum number of minutes old that the
            data can be.
        @param last_update: datetime object representing when it was last
            updated
        @param current: dictionary representing Current json object from
            the api response
        """
        self.tolerance = tolerance
        self.last_update = last_update
        self.current = current
        last_update_str = self.last_update.strftime('%y-%m-%d %H:%M')
        self.message = ('update at {} UTC exceeds tolerance of {} '
                        'minutes'.format(last_update_str, tolerance))
        super().__init__(self.message)


class PjmApi():
    """Grab electricity load data from API."""

    def __init__(self, api_key):
        """
        Initialize instance variables.

        @param api_key: API key
        """
        self.API_KEY = api_key
        self.sess = requests.Session()
        self.sess.headers.update(
            {'Ocp-Apim-Subscription-Key': self.API_KEY})

    def get_load(self, area, tolerance=60):
        """
        Get instantaneous load data for the area.

        Gets the instantaneous load data for the area.
        @param area: column in sql loads table that this is for
        @param tolerance: number of minutes old that weather information
            returned is allowed to be.
        @return int number of the instantaneous load
        """
        # Determine tolerance and current time
        tolerance = int(tolerance)  # In case user puts in a float
        tolerance_td = datetime.timedelta(minutes=tolerance)
        now = datetime.datetime.now(datetime.timezone.utc)

        # Request comed information
        api_area = consts.LOAD_CLMN_TO_LOAD_AREA[area]
        try:
            response = self.sess.get(BASE_URL.format(api_area))
        except Exception as e:
            raise RequestError(msg=str(e))
        if response.status_code != 200:
            raise RequestError(http_code=response.status_code)

        # Extract information
        try:
            # Convert to json dictionary
            json = response.json()

            # Check to see if data is too old
            # Ex)  2023-05-12T23:19:58.4
            last_updated_str = json['items'][0]['datetime_beginning_utc']
            last_updated_str = last_updated_str[:16] + 'UTC'
            last_updated = datetime.datetime.strptime(last_updated_str,
                                                      '%Y-%m-%dT%H:%M%Z')
            last_updated = last_updated.replace(tzinfo=datetime.timezone.utc)
            difference = now - last_updated
            if now - last_updated > tolerance_td:
                raise StaleDataError(tolerance, last_updated, json['current'])

            # Return load data
            return int(json['items'][0]['instantaneous_load'])

        except StaleDataError as e:
            # So that this isn't caught and misreported below
            raise e
        except Exception:
            raise UnexpectedResponseError(text=response.text)


if __name__ == '__main__':
    import os
    pjm = PjmApi(os.environ['PJM_API_KEY'])
    print(pjm.get_load('comed'))
    print(pjm.get_load('pjm'))
