import configparser
import os

def read_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "aqua.cfg")

    config = configparser.ConfigParser()
    config.read(config_file)

    if "aqua" not in config:
        print("aqua.cfg does not have a [aqua] section.")
        exit(-1)

    config = config["aqua"]
    return config
