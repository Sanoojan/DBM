from models.shallow_net import ShallowNet
from models.sklearn_model import SklearnModel


def load_model(args, dataloader, device):
    if args.model in ["RandomForest", "GradientBoosting", "SVM", "LogisticRegression"]:
        return SklearnModel(args, dataloader, device)
    elif args.model == "ShallowNet":
        return ShallowNet(args, dataloader, device)
    else:
        raise Exception(f"Model {args.model} does not exist")
