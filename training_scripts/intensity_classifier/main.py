from utils.train_test_splitGen import DatasetLoader
from utils.dataset import MHCoPilot_Dataset
from torch.utils.data import DataLoader
import torch
from models.clf_RoBERTa.model import RobertaForMultiLabelMulticlassClassification, RobertaForMultiLabelMulticlassClassificationUtils
from utils.trainer import epoch_controller
from utils.load_model import load_model
from transformers import RobertaConfig
import wandb

torch.cuda.empty_cache()
wandb.login()

hyperparameters = dict(
    RANDOM_SEED = 42,
    BATCH_SIZE = 32,
    EPOCHS = 10,
    PATIENCE = 5,
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    LOAD_MODEL_CHECKPOINT = False,
    MAKE_NEW_SPLIT = False,
    MODEL_CHECKPOINT_PATH = './models/checkpoints/clf_RoBERTa.pt',
    ARTITECTURE = 'RoBERTa-cross_entropy',
    RUN_COUNT = 1,
    TASK = 'classification'
)

with wandb.init(project="MHCoPilot_"+hyperparameters["TASK"], name=hyperparameters["ARTITECTURE"]+"_"+str(hyperparameters["RUN_COUNT"]), config=hyperparameters, settings=wandb.Settings(symlink=False), reinit=True):
    config = wandb.config

    datasets = DatasetLoader('../data')
    datasets.make_train_test_split(make_new_split=hyperparameters["MAKE_NEW_SPLIT"])

    model_config = RobertaConfig.from_pretrained(
        'roberta-base',
        num_labels=3,  
    )

    model = RobertaForMultiLabelMulticlassClassification.from_pretrained(
        'roberta-base',
        config=model_config,
        num_labels=3,
        num_classes_per_label=3
    )

    if(config['LOAD_MODEL_CHECKPOINT']):
        model = load_model(model, config['MODEL_CHECKPOINT_PATH'])
    model_util = RobertaForMultiLabelMulticlassClassificationUtils(model,config['ARTITECTURE'], config['DEVICE'])
    model.resize_token_embeddings(len(model_util.tokenizer))

    train_dataset = MHCoPilot_Dataset(datasets.train_df, config["TASK"], model_util.tokenizer)
    val_dataset = MHCoPilot_Dataset(datasets.val_df, config["TASK"], model_util.tokenizer)
    test_dataset = MHCoPilot_Dataset(datasets.test_df, config["TASK"], model_util.tokenizer)


    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    epoch_controller = epoch_controller(model, model_util, train_loader, val_loader, test_loader, epochs = config['EPOCHS'], device = config["DEVICE"], early_PATIENCE = config['PATIENCE'])
    epoch_controller.run()