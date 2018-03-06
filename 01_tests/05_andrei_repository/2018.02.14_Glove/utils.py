import os
import io
import locale

def guess_encoding(file_path):
    with io.open(file_path, "rb") as f:
        data = f.read(5)
    if data.startswith(b"\xEF\xBB\xBF"):  # UTF-8 with a "BOM"
        return "utf-8-sig"
    elif data.startswith(b"\xFF\xFE") or data.startswith(b"\xFE\xFF"):
        return "utf-16"
    else:  # in Windows, guessing utf-8 doesn't work, so we have to try
        try:
            with io.open(file_path, encoding="utf-8") as f:
                preview = f.read(222222)
                return "utf-8"
        except:
            return locale.getdefaultlocale()[1]

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    logger_lib = None
    print("Logger library not found in shared repo.")
    #raise Exception("Couldn't find google drive folder!")
  else:
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)), flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)), flush = True)

  return logger_lib

class SimpleLogger:
  def __init__(self):
    return
  def VerboseLog(self, _str, show_time):
    print(_str, flush = True)

def LoadLogger(lib_name, config_file, log_suffix = "", TF_KERAS = False, HTML = False):
  module = load_module("logger", "logger.py")
  if module is not None:
    logger = module.Logger(lib_name = lib_name,
                           config_file = config_file,
                           log_suffix = log_suffix,
                           TF_KERAS = TF_KERAS,
                           HTML = HTML)
  else:
    logger = SimpleLogger()
  return logger
