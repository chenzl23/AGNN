import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--dataset-name", nargs = "?", default = "ACM")
    parser.add_argument("--epoch-num", type = int, default = 500, help = "Number of training epochs. Default is 200.")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed for train-test split. Default is 42.")
    parser.add_argument("--dropout", type = float, default = 0.5, help = "Dropout parameter.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate. Default is 0.01.")

    parser.add_argument("--lr-patience", type = int, default = 40, help = "Patience for learning rate adapt")

    # Dataspilt
    parser.add_argument("--num-train-per-class", type = int, default = 20, help = "Train data num. Default is 20.")
    parser.add_argument("--num-val", type = int, default = 500, help = "Valid data num. Default is 500.")
    parser.add_argument("--num-test", type = int, default = 1000, help = "Test data num. Default is 1000.")
    parser.add_argument("--feature-normalize", type = int, default = 1, help = "If feature normalization")

    # Parameters
    parser.add_argument("--lamda", type = float, default = 0.9, help = "Lipz constant.")
    parser.add_argument("--theta1", type = float, default = 0.02, help = "Theta constant.")
    parser.add_argument("--theta2", type = float, default = 0.04, help = "Theta constant.")
    parser.add_argument("--rho", type = float, default = 0.1, help = "Parameter for Adaboost.")
    
    parser.add_argument("--layer-num", type = int, default = 4, help = "Layer number.")
    parser.add_argument("--hidden-dim", type = int, default = 128, help = "Hidden dimension.")

    # for early stop
    parser.add_argument("--early-stop", type = bool, default = False, help = "If early stop")
    parser.add_argument("--patience", type = int, default = 20, help = "Patience for early stop")
    
    return parser.parse_args()
