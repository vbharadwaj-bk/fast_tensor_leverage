import numpy as np
from datetime import datetime
import subprocess
import json
import os

class Experiment:
    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.data["omp_threadcount"] = os.environ["OMP_NUM_THREADS"]
        self.data["date"] = str(datetime.now())
        result = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True) 
        self.data["commit_hash"] = result.stdout.strip() 

    def get_data(self):
        return self.data

    def write_to_file(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)