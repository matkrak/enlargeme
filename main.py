import os
import time
from datetime import datetime
import logging

from tests import *

## SET LOGGING LEVEL HERE !
LOGGING_LEVEL = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)


def handleLogFile(filename, choice=None, loggingLevel=LOGGING_LEVEL):
    '''Handle file for logging purpose.
    '''

    if choice not in (None, 1, 2, 3):
        raise ValueError('If choice parameter is provided, should be 1, 2 or 3')

    pardir = os.path.abspath(os.path.join(filename, os.pardir))
    print('Path to log directory: ' + pardir)
    fname = filename.split(os.sep)[-1]

    if fname in os.listdir(pardir):
        if choice is None:
            print('Logfile ' + filename + ' already exists.\n'
                    '[1] Backup as ' + fname + '.old (overwrite if necessary)\n'
                    '[2] Remove existing file\n'
                    '[3] Continue writing to ' + fname + '\n')

            while True:
                try:
                    choice = input()
                    if choice in ('1', '2', '3'):
                        break
                except (EOFError, NameError, SyntaxError):
                    continue

        if choice == '1':
            os.rename(filename, filename + '.old')
            print('Successfully created backup :' + filename + '.old')
        elif choice == '2':
            try:
                os.remove(filename)
                print('Successfully removed old file.')
            except OSError:
                print('File ' + filename + ' does not exist so can\'t be removed! Logging to: ' + filename)
        else:
            print('Continue logging to previous log file : ' + filename)

    # if choice in (1, 2) or len(log.handlers) == 0:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    handler.setLevel(loggingLevel)
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.addHandler(handler)

    print('Logging to: ' + filename)
    print('handleLogFile' + '.' * 30 + 'DONE\n')

    st = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger.info('*' * 80)
    logger.info('Started logging, current time: ' + st)


if __name__ == '__main__':
    print('Tell me what to do!')
    #compareFiltersNo()
    #compareMatrixScalar()