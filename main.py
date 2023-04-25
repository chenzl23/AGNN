import torch
from paraparser import parameter_parser
from utils import tab_printer
from model import Net
from train import train
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import random
from DataLoader import load_data


def main():
    args = parameter_parser()
    tab_printer(args)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    graph = load_data(args)
    number_class = torch.unique(graph.y).shape[0]  
    args.num_class = number_class
    processed_dir = os.path.join(os.path.join(os.path.join("../data",args.dataset_name), args.dataset_name), "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    print(f"Data statistics:  #features {graph.x.size(1)}, #nodes {graph.x.size(0)}")


    dataloader = graph


    input_channels = graph.x.size(1)
    output_channels = len(torch.unique(graph.y))

    model = Net(args, input_channels, output_channels, graph.x.size(0), device).to(device)

    graph = graph.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=5e-4)

    train(model, optimizer, dataloader, graph, args, device)



if __name__ == "__main__":
    main()

