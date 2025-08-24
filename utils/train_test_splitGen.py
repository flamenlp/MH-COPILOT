import pandas as pd
from sklearn.model_selection import train_test_split
import os 

TRAIN_VAL_TEST_SPLIT_PATH = 'train_val_test_split'

def generate_label_combinations(row):
    # print(row['ES'], row['EFS'], row['RS'])
    # print(row)
    return str(int(row['ES'])) + str(int(row['EFS'])) + str(int(row['RS']))

class DatasetLoader:
    def __init__(self, path):
        self.path = path
        self.mhcp_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data(self):
        sheets = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        for sheet in sheets:
            if(sheet[-5:] != '.xlsx'):
                continue
            current_sheet_df = pd.read_excel(os.path.join(self.path, sheet))
            if self.mhcp_df is None:
                self.mhcp_df = current_sheet_df
            else:
                self.mhcp_df = pd.concat([self.mhcp_df, current_sheet_df], axis=0)

    def preprocess_data(self):
        # Temporary Preprocessing Steps
        self.mhcp_df = self.mhcp_df.loc[self.mhcp_df['Annotated'] == True]
        # self.mhcp_df = self.mhcp_df.loc[self.mhcp_df['Comments'] != 'random']
        # self.mhcp_df = self.mhcp_df.loc[self.mhcp_df['Comments'] != 'positive']

        # due to tokenizer issues
        self.mhcp_df['annotated_post_body'] = self.mhcp_df['annotated_post_body'].replace('\n',' ',regex=True)

        self.mhcp_df.drop(columns=['Unnamed: 0', 'Random number', 0], inplace=True)
        self.mhcp_df = self.mhcp_df.reset_index(drop=True)

    def get_dataframe(self):
        return self.mhcp_df 

    def make_train_test_split(self, val_split = 0.2, test_split = 0.1, make_new_split = True):
        self.load_data()
        self.preprocess_data()
        self.mhcp_df['label_combination'] = self.mhcp_df.apply(generate_label_combinations, axis=1)
        self.mhcp_df['label_combination'].value_counts().plot(kind='bar',ylabel="Number of posts").get_figure().savefig('label_combination_distribution_bar.png')
        # stratified splitting of the df
        if os.path.exists(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH)) and not make_new_split:
            self.train_df = pd.read_csv(os.path.join(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), 'train.csv'))
            self.val_df = pd.read_csv(os.path.join(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), 'val.csv'))
            self.test_df = pd.read_csv(os.path.join(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), 'test.csv'))
        else:
            os.makedirs(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), exist_ok=True)
            y = self.mhcp_df['label_combination']
            X_train_val, X_test, y_train_val, _ = train_test_split(self.mhcp_df, y, test_size=test_split, stratify=y, random_state=42)
            X_train, X_val, _, _ = train_test_split(X_train_val, y_train_val, test_size=val_split/(1-test_split), stratify=y_train_val, random_state=42)
            self.train_df = X_train
            self.val_df = X_val
            self.test_df = X_test
            self.train_df.to_csv(os.path.join(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), 'train.csv'), index=False)
            self.val_df.to_csv(os.path.join(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), 'val.csv'), index=False)
            self.test_df.to_csv(os.path.join(os.path.join(self.path, TRAIN_VAL_TEST_SPLIT_PATH), 'test.csv'), index=False)
        
    