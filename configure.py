import os
import shutil
import subprocess
import sys

if __name__=='__main__':
	files = os.listdir()
    
	if "env.sh" not in files:
		shutil.copy("env_template.sh", "env.sh") 

	if "config.json" not in files:
		shutil.copy("config_template.json", "config.json") 
