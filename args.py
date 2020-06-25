import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
                        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="model running mode (train/valid/test)")
    parser.add_argument("--cycle", action="store_true",
                        help="enable cyclic training")
    parser.add_argument("--distil", action="store_true",
                        help="enable distillation training")
    parser.add_argument("--valid", action="store_true",
                        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")
    parser.add_argument("--tta", action="store_true",
                        help="enable tta infer")
    parser.add_argument("--swa", action="store_true",
                        help="finetune swa cycle")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="enable debug mode for test")

    args = parser.parse_args()
    if args.cycle:
        args.mode = "cycle"
    elif args.distil:
        args.mode = "distil"
    elif args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"
    elif args.swa:
        args.mode = "swa"

    return args
