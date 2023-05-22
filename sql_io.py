#!/usr/bin/python
"""
Purpose: Serve as an interface with a SQL database

This will help with inserts, deletes, and csv generations
of the data stored in SQL.

Intended usage here will be with Sqlite3 database.
"""
import consts
import copy
import datetime
import os
import sqlalchemy
import consts


class ExistentialError(Exception):
    """Raise when trying to create new database and file already exists."""
    def __init__(self, msg=None):
        """Initialize error."""
        self.message = msg if msg else 'ran into problem with db structure'
        super().__init__(self.message)


class SqlDbError(Exception):
    """Raise when encountering errors interacting with the database."""
    def __init__(self, e):
        """
        Initialize error.

        @param e: the sqlalchemy.exc.* object
        """
        self.e = e
        self.message = e.orig
        super().__init__(self.message)


class SqlIO():
    """Facilitate SQL operations."""

    def __init__(self, path_to_db, create=False, echo=False):
        """
        Initialize instance fields.

        @param path_to_db: string path to the database
        @param create: whether or not to create a database if it does not
            exist. If True, this will update the schema of any existing
            database at the specified path. If False, this raises an error
        @param echo: if all sql operations are logged and printed to
            stdout (Both)
        """
        # Save list of valid areas
        self.areas = set([area[0] for area in consts.AREAS])

        # Check to see if database exists if create=False
        self.path_to_db = path_to_db
        if not create and not os.path.isfile(path_to_db):
            raise ExistentialError('database not found at {}'.format(
                path_to_db))

        # Create engine to interface with database
        self.eng = sqlalchemy.create_engine('sqlite:///{}'.format(
            path_to_db), echo=echo)

        # Create db if necessary
        self.verify_db(create)

    def insert_load_sample(self, sample_time, values):
        """
        Insert a sample of comed and pjm loads.

        Sample times are stored in the database with accuracy to the second.
        This will insert the sample's time into the samples table if it is
        missing and the comed and pjm loads into the loads table.
        @param sample_time: datetime object for when the sample was recorded
        @param values: dict of the electricity load (MW) as values, grid
            source as the key as defined in consts.LOADS_TABLE_COLUMNS
            (Ex: {'comed': 9000})
        """
        values = copy.deepcopy(values)

        # Verify parameter types
        if not isinstance(sample_time, datetime.datetime):
            raise TypeError('sample_time must be datetime object')

        # Get sample id from samples table
        unix_timestamp = int(sample_time.timestamp())
        sample_id = self.get_sample_id(unix_timestamp)
        values['sample_id'] = sample_id

        # Insert it into table
        self.insert_rows('loads', [values])

    def insert_location_sample(self, sample_time, location, values):
        """
        Insert a sample into a location table.

        @param sample_time: datetime object for when the sample was recorded
        @param location: the location for this sample
        @param values: dict mapping each column to its value to insert.
            Columns and values must match what's defined in
            consts.LOCATION_TABLE_COLUMNS
        """
        values = copy.deepcopy(values)

        # Verify parameter types
        if not isinstance(sample_time, datetime.datetime):
            raise TypeError('sample_time must be datetime object')
        if location not in self.areas:
            raise ValueError('invalid location {}'.format(location))
        for column in values:
            if column not in consts.LOCATION_TABLE_COLUMNS:
                raise ValueError('invalid column {} for location '
                                 'table'.format(column))

        # Get sample id from samples table
        unix_timestamp = int(sample_time.timestamp())
        sample_id = self.get_sample_id(unix_timestamp)
        values['sample_id'] = sample_id

        # Insert it into table
        self.insert_rows(location, [values])

    def get_sample_id(self, unix_timestamp):
        """
        Get the sample id for the unix_timestamp.

        If the timestamp is missing from the table, this inserts it as a
        new row.
        @param unix_timestamp: int of unix_timestamp for the sample
        @return the id for the row containing the unix_timestamp
            in the samples table
        """
        # Query for any existing sample with matching timestamp
        unix_timestamp = int(unix_timestamp)
        query = ('select id from samples where unix_timestamp = '
                 '"{}"'.format(unix_timestamp))
        results = self.execute(query).fetchall()
        if len(results) > 0:
            return results[0][0]

        # Insert timestamp
        create_string = ('insert into samples (unix_timestamp) '
                         'values({})'.format(unix_timestamp))
        self.execute(create_string)

        # Try to get sample id again
        return self.execute(query).fetchall()[0][0]

    def insert_rows(self, table, rows):
        """
        Insert rows into a table.

        @param table: string of the table name
        @param rows: list of dicts. Each dict has the keys as the columns
            and the values as the items to insert into the table.
        """
        # Verify types of rows
        if not (isinstance(rows, list) or isinstance(rows, tuple)):
            raise TypeError('rows must be list-like')
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError('each row must be a dict of each column '
                                'and value for that column')

        # Perform inserts
        for row in rows:
            columns = []
            values = []
            for column in row:
                columns.append(column)
                values.append(str(row[column]))
            insert_string = 'insert into {}({}) values({})'.format(
                table,
                ','.join(columns),
                ','.join(values))
            self.execute(insert_string)

    def verify_db(self, create):
        """
        Verify database design.

        Verifies that all expected tables and columns are in the database.
        If any unexpected columns are encountered or any expected columns
        are missing, this will raise ExistentialError.
        If any tables are missing and create=True, this will create those
        tables. If a table is missing and create=False, this will raise
        ExistentialError.
        @param create: True if this script can create tables, else False
        """
        # Determine tables and columns we are looking for
        expected_tables = {'samples', 'loads'} | self.areas
        expected_area_columns = set(consts.LOCATION_TABLE_COLUMNS.keys())
        expected_load_columns = set(consts.LOADS_TABLE_COLUMNS.keys())
        expected_sample_columns = set(consts.SAMPLES_TABLE_COLUMNS.keys())

        # Check for all tables and columns
        inspector = sqlalchemy.inspect(self.eng)
        tables = inspector.get_table_names(schema='main')
        for table in tables:
            if table not in expected_tables:
                # Not a problem, ignore
                continue

            # Remove table from expected_tables to keep track of tables
            # still missing
            expected_tables.remove(table)

            # Determine which columns we expect from this table
            if table in self.areas:
                expected_columns = expected_area_columns
            elif table == 'samples':
                expected_columns = expected_sample_columns
            else:
                expected_columns = expected_load_columns

            # Verify correct columns
            found_columns = set()  # Keep track of expected columns we find
            for column in inspector.get_columns(table, schema='main'):
                column_name = column['name']
                if (column_name not in expected_columns
                        and column_name != 'id'):
                    raise ExistentialError('unexpected column {} found '
                                           'in {}'.format(column_name,
                                                          table))
                found_columns.add(column_name)
            missing_columns = list(expected_columns - found_columns)
            if len(missing_columns) > 0:
                # Missing at least one column from this table
                raise ExistentialError('missing column {} from table '
                                       '{}'.format(missing_columns[0],
                                                   table))

        # Create missing tables
        if not create and len(expected_tables) > 0:
            # Not allowed to create, raise issue
            raise ExistentialError('missing tables {}, create=False'.format(
                expected_tables))

        for table in expected_tables:

            # Determine columns to use
            if table in self.areas:
                columns = consts.LOCATION_TABLE_COLUMNS
            elif table == 'samples':
                columns = consts.SAMPLES_TABLE_COLUMNS
            else:
                columns = consts.LOADS_TABLE_COLUMNS

            # Create table
            create_string = ('create table {}(id integer primary key '
                             'autoincrement'.format(table))
            for column in columns:
                create_string += ', {} {}'.format(
                    column,
                    columns[column]['datatype'])
            create_string += ')'
            self.execute(create_string)

    def execute(self, query):
        """
        Execute the sql query.

        @param query: sql query string (excluding final semicolon)
        @return sqlalchemy.engine.cursor.CursorResult object of results
        """
        with self.eng.connect() as conn:
            try:
                results = conn.execute(sqlalchemy.text(query))
                conn.commit()
            except Exception as e:
                raise SqlDbError(e)
        return results


if __name__ == '__main__':
    file_path = 'weatherData.db'
    db = SqlIO(file_path, create=True)
    sample_time = datetime.datetime(1999, 5, 7, 6, 10)
    print(sample_time)
    print('tz: {}'.format(sample_time.tzinfo))
    #db.insert_load_sample(sample_time, {"comed": 50, "pjm": 100})
    db.insert_location_sample(sample_time, 'nyc', {"temperature": 50, "humidity": 100})
