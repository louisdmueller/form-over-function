import csv
import logging
import os


def setup_logger(logger_name: str, log_level: int) -> logging.Logger:
    """Set up a logger that logs to both console and a file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    logger.debug(f"Logger '{logger_name}' set up with level {logging.getLevelName(log_level)}")
        
    return logger

def log_job_info(
    tasks: dict, config: dict, execution_time: int
) -> None:
    # append job information to a csv file
    # TODO: ressources used, files written, models used, ...
    job_info = {
        "job_id": os.getenv("SLURM_JOB_ID", "local_run"),
        "execution_time": execution_time,
        "base_data_model": tasks["base_data_model"],
        "judge_model": tasks["judge_model_name"],
    }

    csv_file_path = config["run_info_csv_path"]
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as csvfile:
        fieldnames = list(job_info.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(job_info)