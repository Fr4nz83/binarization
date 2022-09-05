from sklearn.model_selection import ParameterGrid
import random
from datetime import datetime
from trainer import train_model
import pandas as pd
import os
from utils.arg_parser import get_parser


def main():

    args = get_parser().parse_args()
    params = {}
    params['epochs'] = [100]
    params['lr'] = [0.01, 0.001]
    params['weight_decay'] = [1e-6, 0]
    params['optim'] = ["adam"]

    params['max_grad_norm'] = [None]
    #params['batch_size'] = [512]
    #params['optim'] = ["adam"]
    #params['scheduler'] = ["cos", "mstep"]
    columns = list(params.keys())
    columns.append('Top1')
    df_log = pd.DataFrame(columns=columns)

    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name + "__" + datestring
    root_log_dir = os.path.join(args.output_dir, name_dir)
    os.makedirs(root_log_dir)
    print("Root Log Dir", root_log_dir)

    print("Launching the grid search with: ")
    print(params)

    # skipping_conf = [{"lr": 0.1, "weight_decay": 0.001, "weight_decay_apq": 0, "batch_size": 128},
    #                  {"lr": 0.1, "weight_decay": 0.001, "weight_decay_apq": 1e-05, "batch_size": 128},
    #                  {"lr": 0.1, "weight_decay": 0.001, "weight_decay_apq": 1e-06, "batch_size": 128}]

    skipping_conf = []

    for param_conf in ParameterGrid(params):
        if not param_conf in skipping_conf:
            print(param_conf)
            log_dir_name = ""
            for key in params.keys():
                setattr(args, key, param_conf[key])
                log_dir_name += "_" + key + "_" + str(param_conf[key])
            #log_dir = os.path.join(root_log_dir, log_dir_name)
            args.name = log_dir_name
            args.output_dir = root_log_dir
            try:
                prec1 = train_model(args)
            except Exception as e:
                print(e)
                prec1 = 0

            df_line = param_conf
            df_line["Top1"] = prec1
            current_df = pd.DataFrame([df_line.values()], columns=df_line.keys())
            df_log = df_log.append(current_df)

    csv_path = os.path.join(root_log_dir, "overall_log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
