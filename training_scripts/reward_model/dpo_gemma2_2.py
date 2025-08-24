import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# %env WANDB_WATCH=all
base_model = "google/gemma-2b-it"
sft_model = "google/gemma-2b-it_wo_tax-1/checkpoint-3331"
new_model = "gemma2-dpo-3"
save_options="gemma2_2"
save_options_epochs=50
epochs=1
lr=2e-5
testsize=None
testval=None
eval_batch=10
gen_batch_size=16
run_type="train"

import warnings
warnings.filterwarnings("ignore")
from transformers import (
    AutoModelForCausalLM,
    
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from peft import PeftModelForCausalLM, get_peft_config
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer,AutoModelForCausalLMWithValueHead, setup_chat_format
import prompts
import bitsandbytes as bnb
from trl import DPOConfig, DPOTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
prompt_template=prompts.gemma_get_prompt

from huggingface_hub import login
import json
from torch.utils.data import Dataset as Ds
import gc

import numpy as np
import verifier_module_final_dpo_updated as verifier

token=""
login(token = token)
wandb_api_key = ""
wandb.login(key=wandb_api_key)
run = wandb.init(
    project='huggingface', 
    job_type="training", 
    anonymous="allow",
    name=new_model,
)

torch_dtype = torch.bfloat16
attn_implementation = "flash_attention_2"

def load_infer(base_model,model_name,device):
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side="left"
    tokenizer.padding_side="left"
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # return_dict=True,
    load_in_8bit=True,
    torch_dtype = torch.bfloat16,
    attn_implementation=attn_implementation,
    device_map=device
    )
    peft_model = PeftModel.from_pretrained(model, model_name)
    return peft_model,tokenizer

def load_model(base_model,model_name,device):
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side="left"
    tokenizer.padding_side="left"
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # return_dict=True,
    load_in_8bit=True,
    torch_dtype = torch.bfloat16,
    attn_implementation=attn_implementation,
    device_map=device
    )
    peft_model = PeftModel.from_pretrained(model, model_name+"/train1")
    return peft_model,tokenizer

def load_dpo_model(base_model,model_name,device):
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side="left"
    tokenizer.padding_side="left"
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # return_dict=True,
    use_cache=False,
    attn_implementation=attn_implementation,
    load_in_8bit=True,
    torch_dtype = torch.bfloat16,
    device_map=device
    )
    peft_model = PeftModel.from_pretrained(model, model_name,adapter_name="train1",use_gradient_checkpointing=True,is_trainable=True)
    peft_model.load_adapter(model_name, adapter_name="reference")
    return peft_model,tokenizer



def remove_ques(row):
    if str(row['ES'])==str(2):
        row['EMaskingQ']=""
    return row


from utils.hfDataset import MHCoPilot_Dataset 
Dataset = MHCoPilot_Dataset("../data", task = "general", ner = False, make_new_split = False)
Dataset.get_data()
Dataset.train_df=Dataset.train_df.map(remove_ques)
Dataset.val_df=Dataset.val_df.map(remove_ques)
Dataset.test_df=Dataset.test_df.map(remove_ques)



samples_len=50
system_prompt='''A support seeker on a peer-to-peer (P2P) Online Mental Health Platform (OMHP) is an individual who utilizes digital services to seek assistance/help for managing and improving their mental health, typically through interactions with peer groups or self-help resources.

The parameters are defined as follows:
    Event: This parameter encapsulates the specific situation, activity, or event that is the focal point of the support seeker’s concern. The explicit detailing of such events provides a contextual background essential for empathetic understanding.
    Effect: This aspect targets the impact or consequences of the identified event on the support seeker. By elucidating the effect, the post conveys the emotional or practical repercussions of the event, thereby inviting more targeted and empathetic responses. 
    Requirement: This parameter is critical in directing the nature of the assistance sought. It ranges from emotional and informational support to instrumental aid, thereby guiding the potential response trajectory.

 In the posts on OMHP these parameters can have intensity ranging from 0 to 2, where 0 means absent, 1 means present but needs clarification and 2 being well described based on the presence of these parameters in the post

    Consider the following post by a support seeker on a OMHP, in which the spans of text representing Event, Effect and Requirement have been marked. Also, the intensity levels for each of the parameters in the post have been provided along with the post.
    The post is context of the victim.
    The post <es> and <ee> tags encapsulate the spans for the Event parameter, <efs> and <efe> tags encapsulate the spans for the Effect parameter, and <rs> and <re> tags encapsulate the spans for the Requirement parameter.
'''
def format_chat_template(row):
    post=row['annotated_post_body']
    with open("test.json","r") as f:
        d=json.load(f)
    q=None
    for j in d:
        if str(j['event_scale'])==str(row['ES']) and str(j['effect_scale'])==str(row['EFS']) and str(j['req_scale'])==str(row['RS']):
            q=j
            break
    # row['schema']=json.dumps(q)
    note="Generate 3 questions following the schema according to the scale of event, effect and requirement provided below the post, for helping the support giver to understand more about the victim.Strictly follow the question format of schema.Give only the json output as specified in the schema and no explanation needed."
    prompt="Post: "+post+"\n\n"+"event scale: "+str(row['ES'])+" effect scale: "+str(row['EFS'])+" requirement scale: "+str(row['RS'])+"\n\n"+"schema:\n"+"{\n"+"event_question: "",\n"+"effect_question: "",\n"+"requirement_question: ""\n}"+"\n\n"+note
    answer="{"+"\n"+"event_question: "+str(row['EMaskingQ'].replace("X",row["EMask"]))+",\n"+"effect_question: "+str(row['EFSMaskingQ'].replace("X",row["EFSMask"]))+",\n"+"requirement_question: "+str(row['RMaskingQ'].replace("X",row["RMask"]))+"\n"+"}"
    row['ES']=int(row['ES'])
    row['EFS']=int(row['EFS'])
    row['RS']=int(row['RS'])
    row['query']=prompt_template(system_prompt,prompt)
    row['answer']=answer
    return row

def prepare_eval(row):
    post=row['annotated_post_body']
    note="Generate 3 questions following the schema according to the scale of event, effect and requirement provided below the post, for helping the support giver to understand more about the victim.Strictly follow the question format of schema.Give only the json output as specified in the schema and no explanation needed."
    prompt="Post: "+post+"\n\n"+"event scale: "+str(row['ES'])+" effect scale: "+str(row['EFS'])+" requirement scale: "+str(row['RS'])+"\n\n"+"schema:\n"+'''{"event_question": "","effect_question": "","requirement_question": ""}'''+"\n\n"+note
    answer=f'''{{"event_question": "{str(row['EMaskingQ'].replace("X",row["EMask"]))}","effect_question": "{str(row['EFSMaskingQ'].replace("X",row["EFSMask"]))}","requirement_question": "{str(row['RMaskingQ'].replace("X",row["RMask"]))}"}}'''
    
    # row['test'] = tokenizer.apply_chat_template(row_json, tokenize=False,add_generation_prompt=True)
    # row['test']=f'''<|start_header_id|>system<|end_header_id|>\n\n
    #         { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
    #         { prompt }<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    row['test']=prompt_template(system_prompt,prompt)
    row['answer']=answer
    return row

if testsize!=None:

    train_dataset = Dataset.train_df.select(range(testsize)).map(
        format_chat_template,batched=False,
        # num_proc=4,
    )
    val_dataset = Dataset.val_df.select(range(testsize)).map(
        prepare_eval,batched=False,
        num_proc=4,
    )
    test_dataset = Dataset.test_df.select(range(testsize)).map(
        prepare_eval,batched=False,
        num_proc=4,
    )

else:
    train_dataset = Dataset.train_df.map(
        format_chat_template,batched=False,
        # num_proc=4,
    )
    val_dataset = Dataset.val_df.map(
        format_chat_template,batched=False,
        # num_proc=4,
    )
    test_dataset = Dataset.test_df.map(
        format_chat_template,batched=False,
        # num_proc=4,
    )
if testval:
    test_dataset=test_dataset.select(range(testval))
class CustomDataset(Ds):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict['choice1'])

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.data_dict.keys()}
    

generation_kwargs={
    "num_beams":2,
    "max_new_tokens":100,
    "do_sample":False,
    "num_return_sequences":2
}
def prepare_inputs(train_dataset,device2):
    torch.cuda.empty_cache()
    model,tokenizer=load_infer(base_model,sft_model,"cuda:1")
    val_test=train_dataset.map(prepare_eval,num_proc=4)

    dataloader = DataLoader(val_test, batch_size=gen_batch_size,shuffle=False)
    data={"choice1":[],"choice2":[],"levels":[],"body":[],"prompt":[]}
    steps=0
    for batch in tqdm(dataloader,total=len(dataloader)):
        inputs = tokenizer(batch['test'], return_tensors='pt',padding=True, 
                        max_length=1024,truncation=True).to(device2)
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )
        text = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        x=0
        for i in text:
            if(x%2==0):
                data["choice1"].append(i)
                data["levels"].append([batch['ES'][x//2].item(),batch['EFS'][x//2].item(),batch['RS'][x//2].item()])
                data["body"].append(batch['body'][x//2])
                data["prompt"].append(batch['test'][x//2])
            else:
                data["choice2"].append(i)
            x+=1
        steps+=len(batch['test'])
        tqdm.write(f"Steps: {steps}")
        del inputs,outputs,text
        if steps%save_options_epochs==0:
            with open(f"preds_{save_options}.json","w") as f:
                f.write(json.dumps(data))
    del model,tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return data

# preds=prepare_inputs(train_dataset,"cuda:0")

# with open(f"preds_{save_options}.json","w") as f:
#     f.write(json.dumps(preds))

# with open(f"preds_{save_options}.json","r") as f:
#     preds=json.load(f)
    
with open(f"preds_gemma2.json","r") as f:
    preds=json.load(f)

print(len(preds['choice1']))
input_dataset=CustomDataset(preds)
# verf=verifier.Verifier("roberta","groq","cuda:0",3)

def prepare_train_dataset(input_dataset):
    dataset={'chosen':[], 'rejected':[],"prompt":[],"rew1":[],"rew2":[],"choosed":[]}
    dataloader=DataLoader(input_dataset,batch_size=16,shuffle=False)
    steps=0
    
    for batch in tqdm(dataloader,total=len(dataloader)):
        # if steps>=992:
        #     steps+=len(batch['choice1'])
        #     continue
        rew1,rew2=verf.reward_calculator(batch['choice1'],batch['choice2'],batch['levels'],batch['body'])
        for i in range(len(rew1)):
            # print(rew1[i].item(),rew2[i].item())  
            if rew1[i].item()>=rew2[i].item():
                dataset["chosen"].append(batch['choice1'][i])
                dataset["rejected"].append(batch['choice2'][i])
                dataset["prompt"].append(batch['prompt'][i])
                dataset['rew1'].append(rew1[i].item())
                dataset['rew2'].append(rew2[i].item())
                dataset['choosed'].append(1)
            else:
                dataset["chosen"].append(batch['choice2'][i])
                dataset["rejected"].append(batch['choice1'][i])
                dataset["prompt"].append(batch['prompt'][i])
                dataset['rew1'].append(rew2[i].item())
                dataset['rew2'].append(rew1[i].item())
                dataset['choosed'].append(2)
        steps+=len(batch['choice1'])
        tqdm.write(f"Steps: {steps}")
        if steps%save_options_epochs==0:
            with open(f"{save_options}.json","w") as f:
                f.write(json.dumps(dataset))
        
    return dataset
# options_dataset=prepare_train_dataset(input_dataset)
# with open(f"{save_options}.json","w") as f:
#     f.write(json.dumps(options_dataset))


with open(f"{save_options}.json","r") as f:
    options_dataset=json.load(f)

print(len(options_dataset['chosen']))
from datasets import Dataset
options_dataset_gen=Dataset.from_dict(options_dataset)
if testsize!=None:
    options_dataset_gen=options_dataset_gen.select(range(testsize))
import gc
# del model,tokenizer



from metrics import compute_metrics,compute_metrics2
from transformers import TrainerCallback

def evaluate(model, tokenizer,val_set,**kwargs):
    # print("Evaluation done")
    torch.cuda.empty_cache()
    model.config.use_cache=True
    val_test=val_set.map(prepare_eval,num_proc=4)
    tokenizer.padding_side="left"
    dataloader = DataLoader(val_test, batch_size=eval_batch)
    preds=[]
    for batch in tqdm(dataloader,total=len(dataloader)):
        inputs = tokenizer(batch['test'], return_tensors='pt',padding=True, 
                        max_length=1024,truncation=True).to("cuda:0")

        outputs = model.generate(
            **inputs,
            num_return_sequences=1,
            max_new_tokens=100,
        )
        text = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        for i in text:
            preds.append(i.strip())
            # preds.append(i.split("assistant")[1].strip())
            # print(i.split("assistant")[1].strip())
        del inputs,outputs,text
        torch.cuda.empty_cache()
    print(preds)
    print(val_test['answer'])
    x=compute_metrics(preds,val_test['answer'])
    print(x)
    wandb.log(x)
    x=compute_metrics2(preds,val_test['answer'])
    print(x)
    del preds

if run_type=="train":
    model,tokenizer=load_dpo_model(base_model,sft_model,"cuda:0")
    # ref_model,tokenizer=load_dpo_model(base_model,sft_model,"cuda:0")
    # model.enable_input_require_grads()
    training_args = DPOConfig(
        output_dir=new_model,
        beta=0.39,
        learning_rate=3e-7,
        model_adapter_name="train1",
        ref_adapter_name="reference",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=15,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,
        bf16=False,
        loss_type="hinge",
        num_train_epochs=epochs,
    )
    dpo_trainer = DPOTrainer(
        model,
        # ref_model,
        args=training_args,
        max_length=1024,
        train_dataset=options_dataset_gen,
        tokenizer=tokenizer,  # for visual language models, use tokenizer=processor instead
    )
    dpo_trainer.train()
    dpo_trainer.save_model(new_model+"final")
    del model,tokenizer,dpo_trainer
    gc.collect()
    torch.cuda.empty_cache()

model,tokenizer=load_model(base_model,new_model+"final","cuda:0")
# model,tokenizer=load_model(base_model,sft_model,"cuda:0")
evaluate(model,tokenizer,test_dataset)

# del model,tokenizer
# model,tokenizer=load_infer(base_model,sft_model,"cuda:0")
# evaluate(model,tokenizer,test_dataset)
wandb.finish()


