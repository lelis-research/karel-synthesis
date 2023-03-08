import datetime
import logging
import os
import sys

from config.config import Config


class StdoutLogger:

    @classmethod
    def init_logger(cls, logger_name: str) -> None:

        output_folder = os.path.join('output', Config.model_name)
        os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_folder, 'logs', f'{logger_name}_{timestamp}.txt'), mode='w')
        ]
        
        logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
        logging.getLogger()
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        
        return logging.getLogger()