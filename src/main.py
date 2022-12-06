import os
import yaml
import argparse 

from train_launcher import train_launcher
from Trainer import trainer_test

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-app', type=int, default=None,
                        help='Select one of three modes : 0 for train, 1 for test and 2 to run')
    parser.add_argument('-config', type=str, default="cfg",
                        help='Add path toward config file')
    parser.add_argument('-test', type=bool, default=False,
                        help="boolean, if True activate the test mode")
    args = parser.parse_args()  

    return args
    
    
def main():
    #Retrieve args
    args = parser()
    
    #Retrieve test mode 
    test = args.test

    #Check if the app exist
    app = args.app 
    assert(app == 0 or app == 1 or app == 2)

    #Check if the config file exist
    config_path = args.config 
    assert(os.path.exists(config_path) or test)

    #Retrieve config file
    if not test:
        with open(config_path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)

    if (app == 0):
        if (test):
            print("Start training test mode")
            trainer_test()
        else:
            print("Start training mode")
            train_launcher(config = config_dict)
    if (app == 1):
        print("Start testing mode")
    if (app == 2):
        print("Start running mode")


if __name__ == "__main__":
    main()