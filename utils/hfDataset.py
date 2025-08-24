from datasets import Dataset
from utils.train_test_splitGen import DatasetLoader
import re

def clf_labels_preprocesser(example):
   labels = [0. for i in range(3)]
   entities = ['ES', 'EFS', 'RS']
   for entity in range(len(entities)):
       entity_level = example[entities[entity]]
       labels[entity] = float(entity_level)
   example['labels'] = labels
   return example

def ner_spanpred_tokenize_and_label(example):
    if example['annotated_post_body'] is None or (isinstance(example['annotated_post_body'], str) and example['annotated_post_body'].lower() == 'nan') or (isinstance(example['annotated_post_body'], float) and math.isnan(example['annotated_post_body'])):
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

    for token in pattern.findall(example['annotated_post_body']):
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
    
    example['ner_tokens'] = tokens
    example['ner_labels'] = labels
    return example

def get_indices_of_tags(sample):
    es_indices = [m.start() for m in re.finditer('<es>', sample)]
    ee_indices = [m.start() for m in re.finditer('<ee>', sample)]
    efs_indices = [m.start() for m in re.finditer('<efs>', sample)]
    efe_indices = [m.start() for m in re.finditer('<efe>', sample)]
    rs_indices = [m.start() for m in re.finditer('<rs>', sample)]
    re_indices = [m.start() for m in re.finditer('<re>', sample)]

    es_indices = [x for x in es_indices]
    ee_indices = [x - 4 for x in ee_indices]
    efs_indices = [x for x in efs_indices]
    efe_indices = [x - 5 for x in efe_indices]
    rs_indices = [x for x in rs_indices]
    re_indices = [x - 4 for x in re_indices]
    
    return es_indices, ee_indices, efs_indices, efe_indices, rs_indices, re_indices

def ind_spanpred(example):
    post = example['annotated_post_body']
    es_indices, ee_indices, efs_indices, efe_indices, rs_indices, re_indices = get_indices_of_tags(post)
    if len(es_indices) <= 100:
        es_indices = es_indices + [-1]*(100-len(es_indices))
        ee_indices = ee_indices + [-1]*(100-len(ee_indices))
    else:
        es_indices = es_indices[:100]
        ee_indices = ee_indices[:100]
    if len(efs_indices) <= 100:
        efs_indices = efs_indices + [-1]*(100-len(efs_indices))
        efe_indices = efe_indices + [-1]*(100-len(efe_indices))
    else:
        efs_indices = efs_indices[:100]
        efe_indices = efe_indices[:100]
    if len(rs_indices) <= 100:
        rs_indices = rs_indices + [-1]*(100-len(rs_indices))
        re_indices = re_indices + [-1]*(100-len(re_indices))
    else:
        rs_indices = rs_indices[:100]
        re_indices = re_indices[:100]


    example['es_indices'] = es_indices
    example['ee_indices'] = ee_indices
    example['efs_indices'] = efs_indices
    example['efe_indices'] = efe_indices
    example['rs_indices'] = rs_indices  
    example['re_indices'] = re_indices
    return example

# def preprocess(dataframe):
#     dataframe["title"] = dataframe["title"].fillna('').astype(str)
#     dataframe["body"] = dataframe["body"].fillna('').astype(str)
#     dataframe["annotated_post_body"] = dataframe["annotated_post_body"].fillna('').astype(str)
#     dataframe["ES"] = dataframe["ES"].fillna('').astype(str)
#     dataframe["EMaskingQ"] = dataframe["EMaskingQ"].fillna('').astype(str)
#     dataframe["EMask"] = dataframe["EMask"].fillna('').astype(str)
#     dataframe["EFSMaskingQ"] = dataframe["EFSMaskingQ"].fillna('').astype(str)
#     dataframe["EFSMask"] = dataframe["EFSMask"].fillna('').astype(str)
#     dataframe["RMaskingQ"] = dataframe["RMaskingQ"].fillna('').astype(str)
#     dataframe["RMask"] = dataframe["RMask"].fillna('').astype(str)
#     dataset = Dataset.from_pandas(dataframe[["title", "body", "annotated_post_body", "ES", "EFS", "RS", "EMaskingQ", "EMask", "EFSMaskingQ", "EFSMask", "RMaskingQ", "RMask"]])
#     return dataset
def preprocess(dataframe):
    dataframe["title"] = dataframe["title"].fillna('').astype(str)
    dataframe["body"] = dataframe["body"].fillna('').astype(str)
    dataframe["annotated_post_body"] = dataframe["annotated_post_body"].fillna(dataframe['title']).astype(str)
    dataframe["ES"] = dataframe["ES"].fillna('').astype(str)
    dataframe["EMaskingQ"] = dataframe["EMaskingQ"].fillna('').astype(str)
    dataframe["EMask"] = dataframe["EMask"].fillna('').astype(str)
    dataframe["EFSMaskingQ"] = dataframe["EFSMaskingQ"].fillna('').astype(str)
    dataframe["EFSMask"] = dataframe["EFSMask"].fillna('').astype(str)
    dataframe["RMaskingQ"] = dataframe["RMaskingQ"].fillna('').astype(str)
    dataframe["RMask"] = dataframe["RMask"].fillna('').astype(str)
    dataset = Dataset.from_pandas(dataframe[["title", "body", "annotated_post_body", "ES", "EFS", "RS", "EMaskingQ", "EMask", "EFSMaskingQ", "EFSMask", "RMaskingQ", "RMask"]])
    return dataset

class MHCoPilot_Dataset():
    def __init__(self, path, task, ner = True, make_new_split = True):
        self.path = path
        self.task = task
        self.ner = ner
        self.dataset = DatasetLoader(path)
        self.dataset.make_train_test_split(make_new_split= make_new_split)
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.clf_class2id = {"absent": 0, "present": 1, "well_described": 2}
        self.clf_id2class = {0: "absent", 1: "present", 2: "well_described"}
        self.ner_spanpred_class2id = {'B-EFS': 3, 'B-ES': 1, 'B-RS': 5, 'I-EFS': 4, 'I-ES': 2, 'I-RS': 6, 'O': 0}
        self.ner_spanpred_id2class = {3: 'B-EFS', 1: 'B-ES', 5: 'B-RS', 4: 'I-EFS', 2: 'I-ES', 6: 'I-RS', 0: 'O'}
    
    def get_data(self):
        self.train_df = preprocess(self.dataset.train_df)
        self.val_df = preprocess(self.dataset.val_df)
        self.test_df = preprocess(self.dataset.test_df)

        if self.task == 'general':
            self.train_df = self.train_df.map(clf_labels_preprocesser)
            self.val_df = self.val_df.map(clf_labels_preprocesser)
            self.test_df = self.test_df.map(clf_labels_preprocesser)

            if self.ner:
                self.train_df = self.train_df.map(ner_spanpred_tokenize_and_label)
                self.val_df = self.val_df.map(ner_spanpred_tokenize_and_label)
                self.test_df = self.test_df.map(ner_spanpred_tokenize_and_label)
            else:
                self.train_df = self.train_df.map(ind_spanpred)
                self.val_df = self.val_df.map(ind_spanpred)
                self.test_df = self.test_df.map(ind_spanpred)
            
        elif self.task == 'classification':
            self.train_df = self.train_df.map(clf_labels_preprocesser)
            self.val_df = self.val_df.map(clf_labels_preprocesser)
            self.test_df = self.test_df.map(clf_labels_preprocesser)

        elif self.task == 'span_prediction':
            if self.ner:
                self.train_df = self.train_df.map(ner_spanpred_tokenize_and_label)
                self.val_df = self.val_df.map(ner_spanpred_tokenize_and_label)
                self.test_df = self.test_df.map(ner_spanpred_tokenize_and_label)
            else:
                self.train_df = self.train_df.map(ind_spanpred)
                self.val_df = self.val_df.map(ind_spanpred)
                self.test_df = self.test_df.map(ind_spanpred)      