import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ['TOKENIZERS_PARALLELISM']="False"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.hfDataset import MHCoPilot_Dataset 
from span_trainer_2 import spanmodel_trainer
from transformers import AutoModelForTokenClassification,T5ForTokenClassification 
from transformers import AutoTokenizer,T5TokenizerFast

id2label = {
    0:'O', 1:'B-ES', 2:'I-ES', 3:'B-EFS', 4:'I-EFS', 5:'B-RS', 6:'I-RS'
}
label2id = {
    'O': 0, 'B-ES': 1, 'I-ES': 2, 'B-EFS': 3, 'I-EFS': 4, 'B-RS': 5, 'I-RS': 6
}

Dataset = MHCoPilot_Dataset("../data", task = "span_prediction", ner = True, make_new_split = False)
Dataset.get_data()
#span_bert
# model_name="distilbert/distilbert-base-cased"
# model_name="/home/karan21258/final_testing/t52/checkpoint-1940/"
# model_name="/home/karan21258/final_testing/t52/checkpoint-1940/"
# model_name="Spanbert/spanbert-base-cased"
model_name="FacebookAI/roberta-base"
# model_name="roberta_large_10_20/checkpoint-1251"
# model_name="google-t5/t5-large"
# model_name="./roberta_large"
dir_name="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
model=AutoModelForTokenClassification.from_pretrained(model_name,num_labels=7, id2label=id2label, label2id=label2id)
spanmodel=spanmodel_trainer(model_name,model,tokenizer,use_cuda=True,context_size=512)
# print(Dataset.train_df)
spanmodel.train(Dataset.train_df,Dataset.val_df,Dataset.test_df,dir_name,batch_size=24,epochs=10,learning_rate=2e-5)
# spanmodel.eval(Dataset.train_df,Dataset.val_df,Dataset.test_df,dir_name,batch_size=12,epochs=10,learning_rate=2e-5)