import hailo_platform
from logger_config import LoggerConfigInfo

logging = LoggerConfigInfo().get_logger(__name__)

def check_hailo_device():
    available_devices = hailo_platform.scan_devices()
    if len(available_devices) == 0:
        logging.error("No Hailo devices found")
        return False
    else:
        logging.info(f"Found {len(available_devices)} Hailo devices")
        return True
