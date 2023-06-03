import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml


def initialize_logger(logfile_name=None, log_level=logging.INFO):
    if logfile_name is None:
        logfile_name = 'entity_recognizer_logs.txt'

    if log_level is None:
        log_level = logging.INFO

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    try:
        if log_level is not None:
            logger.setLevel(log_level)
    except Exception as e:
        logger.setLevel(logging.INFO)
        print(
            f'Error occurred while setting loglevel: {log_level}, falling back to default INFO level, Error: {str(e)}')

    s_handler = logging.StreamHandler()
    s_handler.setLevel(log_level)

    Path(logfile_name).parent.mkdir(parents=True, exist_ok=True)

    f_handler = logging.FileHandler(filename=logfile_name, mode='a')
    f_handler.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s line: %(lineno)d - %(message)s')
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    logger.info(f'Logger initialized, Refer Logfile: {logger.handlers[1].baseFilename}, LogLevel: {logger.level}')
    return logger


def get_logfile_name(config: dict):
    today = datetime.today()
    date_string = f'{today.day}{today.month}{today.year}-{today.hour}{today.minute}{today.second}'
    logfile_name = Path(config["logging_dir"]) / f'{config["logfile_name"]}-{date_string}.txt'

    Path(config["logging_dir"]).mkdir(parents=True, exist_ok=True)

    return logfile_name


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Configuration file', type=str)

    return parser.parse_args()


def load_config(config_file='./config/config.yml'):
    with open(config_file) as f:
        yaml_json = yaml.full_load(f)

    return yaml_json
