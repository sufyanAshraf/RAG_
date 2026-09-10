import logging
 
logger = logging.getLogger()

#formatter
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

# create handler
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler("app.log")

# set formatter
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

#set log level
logger.setLevel(logging.INFO)