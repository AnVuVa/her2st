# Main script to run torch model, use parser for arguments

from sklearn.pipeline import Pipeline


def main(args):
    from dataset import load_and_report_dataset
    from model import AModel
    from pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(epochs=30, batch_size=32, num_workers=8,
                        root_dir="./data", hvg_path="./data/her_hvg_cut_1000.npy",
                        device=args.device)
    pipe = Pipeline(cfg)
    pipe.build_data(fold=args.fold)
    pipe.build_model()
    pipe.build_optimizer()
    if args.phase == "train":
        hist = pipe.fit(fold=args.fold, start_epoch=args.start_epoch)
    elif args.phase == "val":
        hist = pipe.evaluate(fold=args.fold, start_epoch=args.start_epoch, phase="val")
    elif args.phase == "test":
        hist = pipe.evaluate(fold=args.fold, start_epoch=args.start_epoch, phase="test")
    print(hist)

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--phase", type=str, choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
