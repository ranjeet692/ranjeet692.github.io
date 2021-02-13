from flask_failsafe import failsafe
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
log_file_name = os.path.join(base_dir, 'logs/general.log')
logging_level = logging.INFO

#get named logger
logger = logging.getLogger(__name__)

#create handler
handler = TimedRotatingFileHandler(filename=log_file_name, when='M', interval=1, backupCount=5, encoding='utf-8', delay=False)

#create formatter and add to handler
#formatter = Formatter(fmt)
def create_app():
	from uniquemachine_app import app
	return app

if __name__ == "__main__":
	create_app().run(host = '0.0.0.0')

