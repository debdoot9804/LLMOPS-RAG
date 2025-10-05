import os
import logging
from datetime import datetime
import structlog

class Logger:
    def__init__(self,log_dir="logs"):
    self.logs_dir=os.path.join(os.getcwd(),log_dir)
    os.makedirs(self.logs_dir,exist_ok=True)

    log_file=f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.log"
    self.log_path=os.path.join(self.logs_dir,log_file)

    def get_logger(self,name=__file__):
        logger_name=os.path.basename(name)

        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )   
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True
        )

        return structlog.get_logger(logger_name)