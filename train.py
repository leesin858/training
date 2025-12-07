import argparse

from utils.load_from_config import load_from_config

from data import DATA_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yaml", help="Path to the configuration file.")
    return parser.parse_args()



def main():
    args = parse_args()
    args = load_from_config(args.cfg)
    
    dataloader_name = args["dataloder_name"]

    train_dataset = DATA_CONFIG[dataloader_name](args, args["train_dataset"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
    )
    
    



if __name__ == "__main__":
    main()









