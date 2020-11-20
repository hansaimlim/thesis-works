import os
import sys
import re
import pandas as pd
import subprocess
import json

##### JSON modules #####
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data
##### JSON modules #####

def decompress_gzip(zip_file,ext='gz'):
  command="gunzip -f {}".format(zip_file)
  output = subprocess.check_output(['bash','-c',command])
  unzip_file=zip_file.replace(str('.'+ext),'')
  return unzip_file
def compress_gzip(unzip_file,ext='gz',level=9):
  command="gzip -{} {}".format(level,unzip_file)
  output = subprocess.check_output(['bash','-c',command])
  zip_file = unzip_file+"."+ext
  return zip_file

def pandas_df_continuous(data):
  df=pd.DataFrame(data,columns=['InChIKey','UniProt','Activity_type','Relation','Activity_value'])
  return df
def pandas_df_binary(data):
  df=pd.DataFrame(data,columns=['InChIKey','UniProt','Activity'])
  return df

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " Yes/no "
    elif default == "no":
        prompt = " No/yes "
    else:
        raise ValueError("invalid default answer: {}".format(default))

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower().strip().lstrip() #remove beginning and ending spaces, new lines
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
