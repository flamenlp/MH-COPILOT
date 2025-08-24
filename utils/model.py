from transformers import RobertaTokenizer
from transformers import RobertaPreTrainedModel, RobertaModel
from torch.nn import functional as F
from transformers import AdamW
from torch import nn
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RobertaForMultiLabelMulticlassClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_labels=3, num_classes_per_label=3):
        super().__init__(config)
        self.num_labels = num_labels
        self.num_classes_per_label = num_classes_per_label
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels * num_classes_per_label)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels, self.num_classes_per_label)
        return logits

class RobertaForMultiLabelMulticlassClassificationUtils:
    def __init__(self, model,architecture, device = 'cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.metrics = {'Validation loss': None, 'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        self.total_val_loss = 0
        self.all_predictions = []
        self.all_true_labels = []
        self.classwise_metrics = {'Event': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                                  'Effect': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                                  'Requirement': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}}
        self.architecture = architecture
        if self.architecture=='RoBERTa-ordinal_loss':
            self.custom_loss = self.custom_ordinal_loss
        elif self.architecture=='RoBERTa-cross_entropy':
            self.custom_loss = self.custom_cross_entropy_loss
        elif self.architecture=='RoBERTa-mse':
            self.custom_loss = self.mse_loss
    def calculate_metrics(self, batch_count, class_wise_calc = False):
        avg_val_loss = self.total_val_loss / batch_count
        self.all_predictions = np.array(self.all_predictions)
        self.all_true_labels = np.array(self.all_true_labels)
        accuracy = accuracy_score(self.all_true_labels.flatten(), self.all_predictions.flatten())
        precision = precision_score(self.all_true_labels.flatten(), self.all_predictions.flatten(), average='macro', zero_division=0)
        recall = recall_score(self.all_true_labels.flatten(), self.all_predictions.flatten(), average='macro', zero_division=0)
        f1 = f1_score(self.all_true_labels.flatten(), self.all_predictions.flatten(), average='macro', zero_division=0)
        self.metrics = {'loss': avg_val_loss, 'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}
        
        if class_wise_calc:
            for i, label in enumerate(['Event', 'Effect', 'Requirement']):
                label_true = self.all_true_labels[:, i]
                label_pred = self.all_predictions[:, i]
                
                label_accuracy = accuracy_score(label_true, label_pred)
                label_precision = precision_score(label_true, label_pred, average='macro', zero_division=0)
                label_recall = recall_score(label_true, label_pred, average='macro', zero_division=0)
                label_f1 = f1_score(label_true, label_pred, average='macro', zero_division=0)

                self.classwise_metrics[label] = {'accuracy': label_accuracy, 'precision': label_precision, 'recall': label_recall, 'f1': label_f1}
            

    def set_metrics_to_zero(self):
        self.metrics = {'loss': None, 'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        self.total_val_loss = 0
        self.all_predictions = []
        self.all_true_labels = []
        self.classwise_metrics = {'Event': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                                  'Effect': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                                  'Requirement': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}}
    
    def custom_cross_entropy_loss(self, logits, true_indices):
        batch_size, num_labels, _ = logits.size()
        losses = []
        
        for label in range(num_labels):
            label_logits = logits[:, label, :]
            label_true_indices = true_indices[:, label]
            
            label_loss = F.cross_entropy(label_logits, label_true_indices)
            losses.append(label_loss)
        
        total_loss = torch.mean(torch.stack(losses))
        
        return total_loss
    
    def mse_loss(self, logits, true_indices):
        batch_size, num_labels, _ = logits.size()
        losses = []
        num_classes = logits.shape[2] 
        for label in range(num_labels):
            label_logits = logits[:, label, :]
            label_true_indices = true_indices[:, label]
            label_probabilities = F.softmax(label_logits, dim=1)
            one_hot_labels = F.one_hot(label_true_indices, num_classes=num_classes)
            label_loss = F.mse_loss(label_probabilities.float(), one_hot_labels.float())
            losses.append(label_loss)
        
        total_loss = torch.mean(torch.stack(losses))
        
        return total_loss
    
    def ordinal_loss(self, output, target, k):
        loss = torch.mean((output - ((2 * target + 1) / (2 * k)))**2)
        return loss
    
    def custom_ordinal_loss(self, logits, true_indices):
        batch_size, num_labels, _ = logits.size()
        losses = []
        num_classes = logits.shape[2] 
        for label in range(num_labels):
            label_logits = logits[:, label, :]
            label_true_indices = true_indices[:, label]
            label_probabilities = F.softmax(label_logits, dim=1)
            one_hot_labels = F.one_hot(label_true_indices, num_classes=num_classes)
            label_loss = self.ordinal_loss(label_probabilities.float(), one_hot_labels.float(), num_classes)
            losses.append(label_loss)
        
        total_loss = torch.mean(torch.stack(losses))
        
        return total_loss
    
    def trainer(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # loss = self.custom_ordinal_loss(logits, labels.long())
        loss = self.custom_loss(logits, labels.long())
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def evaluator(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.custom_ordinal_loss(logits, labels.long())
        self.total_val_loss += loss.item()
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_true_labels.extend(labels.cpu().numpy())

        return loss.item()




