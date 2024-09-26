import os
import argparse
import time
import numpy as np
import pandas as pd

from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor

parser = argparse.ArgumentParser(
    description="Process Transformer - Data Processing.")

parser.add_argument("--dataset", 
    type=str, 
    default="helpdesk", 
    help="dataset name")

parser.add_argument("--dir_path", 
    type=str, 
    default="./datasets", 
    help="path to store processed data")

parser.add_argument("--raw_log_file", 
    type=str, 
    default="./datasets/helpdesk/helpdesk.csv", 
    help="path to raw csv log file")

parser.add_argument("--task", 
    type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY, 
    help="task name")

parser.add_argument("--sort_temporally", 
    type=bool, 
    default=False, 
    help="sort cases by timestamp")

parser.add_argument("--insert_eot", 
    action="store_true",
    default=False, 
    help="insert end of trace token")

args = parser.parse_args()

if __name__ == "__main__": 
    # Process raw logs
    start = time.time()
    data_processor = LogsDataProcessor(name=args.dataset, 
        filepath=args.raw_log_file, 
        columns = ["Case ID", "Activity", "time:timestamp"], #["case:concept:name", "concept:name", "time:timestamp"], 
        csv_separator=';',
        dir_path=args.dir_path, pool = 1, insert_eot=args.insert_eot)
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally)
    end = time.time()
    print(f"Total processing time: {end - start}")

