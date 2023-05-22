"""
Purpose: This script has the following two objectives:
    1) Continually collect information and store it in a database
    2) Make decisions based on whether or not a peak load hour is upcoming

At this time, only objective 1 is implemented.
"""
import argparse
import consts
import datetime
import logging
import os
import sql_io
import signal
import threading
import pjm_api
import weather_api

# Definitions for argument parser
PROG_NAME = 'Comed Hourly'
DESCRIPTION = ('Collects data and makes predictions on peak PJM and Comed '
               'load hours')

# Outsources sleep to another thread
do_exit = threading.Event()

# Logger and whether or not we print logs
log = None
verbose = False

# Start and end datetime.time objects for every day
start_time = None
end_time = None
# Set of valid days to operate (Abbreviated, ex: Mon, Tue, Wed...)
valid_days = None


# Wrappers for logging
def log_msg(message, level):
    """Log at specified level."""
    log.log(level, message)
    if verbose:
        print(message)


def log_info(message):
    """Log at info level."""
    log_msg(message, logging.INFO)


def log_warn(message):
    """Log at warning level."""
    log_msg(message, logging.WARN)


def log_err(message):
    """Log at error level."""
    log_msg(message, logging.ERROR)


def log_crit(message):
    """Log at critical level."""
    log_msg(message, logging.CRITICAL)


def log_debug(message):
    """Log at debug level."""
    log_msg(message, logging.DEBUG)


def signal_handler(sig, frame):
    """End application."""
    global do_exit
    do_exit.set()


def get_seconds_until_task():
    """
    Determine number of seconds to sleep until next task.

    We want to perform these tasks every 5 minutes, so this will
    calculate the number of seconds until the next 5min interval
    based on the clock (00:05, 00:10, etc.)
    However, the APIs may not be exactly on time, so we add an
    additional 60 seconds as a grace period
    @return seconds to sleep, and datetime object representing next
        interval's time (To the 5 minute mark)
    """
    # Determine seconds until next iteration
    now = datetime.datetime.now()
    secs = (now.minute % 5 * 60) + now.second
    wait_secs = 300 - secs

    # Determine next task's datetime
    # Skip ahead until we are within valid operating hours and day
    plus_5 = datetime.timedelta(minutes=5)
    now = now - datetime.timedelta(microseconds=now.microsecond)
    next_dt = now + datetime.timedelta(seconds=wait_secs)
    next_time = next_dt.time()
    log_debug('picking next time')
    while (next_time < start_time or next_time > end_time
            or next_dt.strftime('%a') not in valid_days):
        next_dt = next_dt + plus_5
        wait_secs += 300  # Add another 5 minutes
        next_time = next_dt.time()

    wait_secs += consts.GRACE_PERIOD  # Add 60 seconds grace period
    return wait_secs, next_dt


def log_inserts(table, inserts):
    """
    Log the data to be inserted into a table

    @param table: name of the table this inserts into
    @param inserts: dict mapping column to the value to be inserted. A
        key may be 'sample_dt' and map to the datetime object for this
        sample instead of the usual column:value pair.
    """
    # Iterate through each column and save its name and value
    columns = []
    row_vals = []
    for column in inserts:
        columns.append(column)
        row_vals.append(str(inserts[column]))

    # Log message
    log_str = '{}\t<<{}={}'.format(table,
                                   ','.join(columns),
                                   ','.join(row_vals))
    log_info(log_str)


def collect_data(wapi, weather_info, papi):
    """
    Collect relevant data.
    
    Grabs the relevant data through APIs and compiles it into a dict and
    subdict, {table: {column: value}}
    @param wapi: WeatherApi object
    @param weather_info: list-like of weather information we want for each
        location from the weather api
    @param papi: PjmApi object
    @return current data as a dict of tables mapping to dicts of their
       mapping to their values. {table: {column: value}}
    """
    # Grab weather data
    # Have each location get its own dict of columns mapping to values
    collected_data = {}
    for area_name, area_cords in consts.AREAS:

        # Query API
        sample_dt = None
        try:
            weather_data = wapi.get_current_weather(info=weather_info,
                                                    latlon=area_cords,
                                                    tolerance=consts.TOLERANCE)
        except weather_api.StaleDataError as e:
            # The API only updates once every 15 minutes, so move this to
            # debug level and attempt to save this under the last_updated
            # datetime in case it was missed before
            log_debug('{} {}'.format(area_name, e.message))
            if e.last_update.minute % 5 == 0:
                collected_data[area_name] = e.ret_info
                collected_data[area_name]['sample_dt'] = e.last_update

        except weather_api.UnexpectedResponseError as e:
            log_err('{}: {}'.format(area_name, e.message))
            for line in e.text.split('\n'):
                if line == '':
                    continue
                log_debug(line)

        except weather_api.RequestError as e:
            log_err('{}: {}'.format(area_name, e.message))

        else:
            # Save info
            collected_data[area_name] = weather_data[0]

    # Grab instantaneous load data
    collected_data['loads'] = {}
    for area in consts.LOAD_CLMN_TO_LOAD_AREA:

        # Query API up to 3 times to get information
        max_tries = 3
        for i in range(0, max_tries):
            try:
                load_data = papi.get_load(area, tolerance=consts.TOLERANCE)

            except pjm_api.StaleDataError as e:
                log_warn('{} {}'.format(area, e.message))
                break

            except (pjm_api.UnexpectedResponseError,
                    pjm_api.RequestError) as e:
                if i + 1 >= max_tries:
                    # Last attempt failed
                    log_err('{}: {}'.format(area, e.message))

                    if isinstance(e, pjm_api.UnexpectedResponseError):
                        for line in e.text.split('\n'):
                            line = line.strip()
                            if line == '':
                                continue
                            log_debug(line)
                else:
                    # More attempts remain
                    log_warn('try {} for {}: {}'.format(i + 1,
                                                        area,
                                                        e.message))
            else:
                # Save info
                collected_data['loads'][area] = load_data
                break

    # Get rid of loads subdict if all load queries failed
    if len(collected_data['loads']) == 0:
        del collected_data['loads']

    # Log findings for each table
    for table in collected_data:
        if 'sample_dt' in collected_data[table]:
            # This is for a previous sample, log later if the insert
            # succeeds
            continue
        log_inserts(table, collected_data[table])

    return collected_data


def main(args):
    # Set verbose
    global verbose
    verbose = args.verbose

    # Set up logger
    global log
    if args.debug:
        # This module and submodules log at debug level
        logging.basicConfig(filename=consts.LOG_FILE,
                            filemode = 'a',
                            format='%(asctime)s.%(msecs)03d '
                                   '[%(levelname)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)
        log = logging.getLogger(__name__)
    else:
        # Make only this module log and at info level
        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)
        fh = logging.FileHandler(consts.LOG_FILE)
        formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d '
                                          '[%(levelname)s]: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        log.addHandler(fh)
    log_info('########## comed_hourlyd started ##########')

    # Create start_time and end_time datetime.time objects
    global valid_days
    global start_time
    global end_time
    if args.anytime:
        # Operate all day, everyday
        valid_days = {'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'}
        start_time = datetime.datetime.strptime('00:00',
                                                '%H:%M').time()
        end_time = datetime.datetime.strptime('23:59',
                                              '%H:%M').time()
    else:
        valid_days = consts.VALID_DAYS
        start_time = datetime.datetime.strptime(consts.START_TIME,
                                                '%H:%M').time()
        end_time = datetime.datetime.strptime(consts.END_TIME,
                                              '%H:%M').time()
    log_debug('logging on days {} between {} and {}'.format(
        valid_days, consts.START_TIME, consts.END_TIME))

    # Create data collectors
    wapi_key = os.environ['WEATHER_API_KEY']
    wapi = weather_api.WeatherApi(wapi_key)
    papi_key = os.environ['PJM_API_KEY']
    papi = pjm_api.PjmApi(papi_key)
    weather_info = list(consts.LOCATION_TABLE_COLUMNS.keys())
    weather_info.remove('sample_id')

    # Sql database
    sql = sql_io.SqlIO(consts.DATABASE,
                       create=args.update_db,
                       echo=args.debug)

    # Handle signals
    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    while not do_exit.is_set():
        # Sleep until next five minute interval
        wait_secs, next_time = get_seconds_until_task()
        log_debug('waiting for {} (unix timestamp {})'.format(
            next_time, next_time.timestamp()))
        do_exit.wait(wait_secs)
        if do_exit.is_set():
            break

        # Collect data
        data = collect_data(wapi, weather_info, papi)

        # Save data to database
        for table in data:

            try:
                if table == 'loads':
                    sql.insert_load_sample(next_time, data[table])

                elif 'sample_dt' in data[table]:
                    # This belongs to a previous sample, try adding and
                    # see if it was missed.
                    missed_dt = data[table]['sample_dt']
                    del data[table]['sample_dt']

                    try:
                        sql.insert_location_sample(missed_dt, table,
                                                   data[table])

                    except sql_io.SqlDbError as e:
                        if not str(e).startswith(
                                'UNIQUE constraint failed'):
                            # This failed for reasons other than being
                            # already inserted
                            raise e

                    else:
                        local_missed_dt = datetime.datetime.fromtimestamp(
                            missed_dt.timestamp())
                        data[table]['sample_dt'] = local_missed_dt.strftime(
                                "'%Y-%m-%d %H:%M'")
                        log_inserts(table, data[table])

                else:
                    sql.insert_location_sample(next_time, table,
                                               data[table])

            except sql_io.SqlDbError as e:
                log_err(e.message)

    log_info("exited gracefully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_NAME,
                                     description=DESCRIPTION)
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true',
                        help='print all log messages')
    parser.add_argument('-d', '--debug', dest='debug',
                        action='store_true',
                        help='sets log level to debug')
    parser.add_argument('-u', '--updatedb', dest='update_db',
                        action='store_true',
                        help='updates database with new tables if any '
                             'are missing')
    parser.add_argument('-a', '--anytime', dest='anytime',
                        action='store_true',
                        help='operates outside of specified days and times')
    main(parser.parse_args())
