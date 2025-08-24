import itertools
import datasets
import logging
from typing import Optional, Dict, Union
from nltk import sent_tokenize
import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import pandas as pd
import re
from transformers import TrainingArguments, Trainer 
from transformers import DataCollatorForTokenClassification 
import os
import numpy as np
label_list=['O', 'B-ES', 'I-ES', 'B-EFS', 'I-EFS', 'B-RS', 'I-RS']
metric = datasets.load_metric("seqeval") 
import wandb

# Set your wandb API key
wandb_api_key = "e62fc492915628e64fcac9c082089ffed84dc72d"

# Log in to wandb using the API key
wandb.login(key=wandb_api_key)


def compute_metrics(eval_preds): 
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2) 
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    
    # We remove all the values where the label is -100
    predictions = [ 
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
    
    true_labels = [ 
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
   ] 
    results = metric.compute(predictions=predictions, references=true_labels) 
    print(results)
    return { 
   "precision": results["overall_precision"], 
   "recall": results["overall_recall"], 
   "f1": results["overall_f1"], 
  "accuracy": results["overall_accuracy"], 
  "complete_results": results
  } 

def tokenize_and_label_ner_final(post_body):
    if post_body is None or (isinstance(post_body, str) and post_body.lower() == 'nan') or (isinstance(post_body, float) and math.isnan(post_body)):
        return [], [], {}

    pattern = re.compile(r"\<\/?\w+\>|\w+|\w+(?:'\w+)?(?:-\w+)*")

    tag_to_label = {
        "<es>": "ES", "<ee>": "/ES",
        "<efs>": "EFS", "<efe>": "/EFS",
        "<rs>": "RS", "<re>": "/RS"
    }

    class_dict = {'O': 0, 'B-ES': 1, 'I-ES': 2, 'B-EFS': 3, 'I-EFS': 4, 'B-RS': 5, 'I-RS': 6}

    tokens, labels = [], []
    current_label = None
    inside_entity = False

    for token in pattern.findall(post_body):
        if token in tag_to_label:
            if token in ["<ee>", "<efe>", "<re>"]:  
                inside_entity = False
            else:  
                inside_entity = True
                current_label = tag_to_label[token]  
            continue

        if inside_entity:
            if current_label and (not labels or labels[-1] == class_dict["O"] or not(labels[-1] == class_dict["B-"+current_label] or labels[-1] == class_dict["I-"+current_label])):
                labels.append(class_dict["B-" + current_label])
            else:
                labels.append(class_dict["I-" + current_label])
        else:
            labels.append(class_dict["O"])
        tokens.append(token)

    return tokens, labels, class_dict

def tokenize_and_align_labels(examples,tokenizer,label_all_tokens=True,context_size=512):
    tokenized_inputs = tokenizer(examples["ner_tokens"],padding='max_length',max_length=context_size, truncation=True, is_split_into_words=True)
    labels = []
    # print(examples["labels"])
    for i, label in enumerate(examples["ner_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class spanmodel_trainer():
    def __init__(
            self, 
            model_name: str,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            context_size: int,
            use_cuda: bool
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        self.context_size=context_size
    
    def __call__(self, inputs):
        print("found")
        inputs = self._prepare_inputs(inputs)
        return inputs


    def _prepare_inputs(self, inputs):
        inputs = inputs.map(lambda examples:tokenize_and_align_labels(examples,self.tokenizer, label_all_tokens=True,context_size=self.context_size), batched=True)
        return inputs
    
    def train(self, train_inputs,val_inputs,test_inputs,dir_name,batch_size=8,epochs=5,learning_rate=2e-3,weight_decay=0.01,evaluation_strategy="epoch"):
        wandb.init(project="huggingface",settings=wandb.Settings(code_dir="."),name=dir_name)
        features=['title', 'body', 'annotated_post_body', 'ES', 'EFS', 'RS', 'EMaskingQ', 'EMask', 'EFSMaskingQ', 'EFSMask', 'RMaskingQ', 'RMask', '__index_level_0__', 'ner_tokens', 'ner_labels']
        train_inputs = self._prepare_inputs(train_inputs)
        val_inputs = self._prepare_inputs(val_inputs)
        test_inputs = self._prepare_inputs(test_inputs)
        # train_inputs = train_inputs.remove_columns(features)
        # val_inputs=val_inputs.remove_columns(features)
        # test_inputs=test_inputs.remove_columns(features)
        print(train_inputs)
        args = TrainingArguments( 
        dir_name,
        evaluation_strategy = "epoch",
        save_strategy="epoch" ,
        # no_cuda=True,
        # warmup_steps=500,  
        learning_rate=learning_rate, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size,
        # place_model_on_device = "cuda", 
        num_train_epochs=epochs, 
        weight_decay=weight_decay, 
        load_best_model_at_end=True, 
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.model.model_parallel = False  
        trainer = Trainer( 
            self.model, 
            args, 
        train_dataset=train_inputs, 
        eval_dataset=val_inputs, 
        data_collator=data_collator, 
        tokenizer=self.tokenizer, 
        compute_metrics=compute_metrics,
        
        ) 
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        trainer.train()
        self.model.save_pretrained(f'./results_{dir_name}')
        self.tokenizer.save_pretrained(f'./results_{dir_name}')
        # trainer.save(f"{self.model_name}-ner-final")
        trainer.evaluate(test_inputs)