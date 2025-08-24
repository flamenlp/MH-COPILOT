import os
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_WATCH']='all'
import prompts
base_model = "google/gemma-2b-it"
new_model = "google/gemma-2b-it_wo_tax-1/checkpoint-3331"
prompt_template=prompts.gemma_get_prompt
epochs=1
lr=2e-5
run_type='evaluate'
load_type='evaluate'
context_length=1024
context_length_eval=1024
eval_batch=10
batch_size=1
testsize=None
new_split=False

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
from tqdm import tqdm
import prompts
from transformers import TrainerCallback
from metrics import compute_metrics
from utils.hfDataset import MHCoPilot_Dataset 



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
# # QLoRA config
# bnb_config = BitsAndBytesConfig(
#     load_in_16bit=True,
#     bnb_16bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )
if run_type=="train":
# # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        load_in_8bit=True,
        use_cache=False,
        device_map="cuda:0",
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True
    )
    # # # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model,truncation_side="left")
    tokenizer.padding_side = 'right'
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, peft_config)


elif load_type=='evaluate':
    tokenizer=AutoTokenizer.from_pretrained(new_model)
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype = torch.bfloat16,
    attn_implementation=attn_implementation,
    device_map="cuda:0"
    )
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # peft_model = PeftModelForCausalLM.from_pretrained(model, sft_model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,new_model)

def remove_ques(row):
    if str(row['ES'])==str(2):
        row['EMaskingQ']=""
    return row


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
    note="Generate 3 questions following the schema according to the scale of event, effect and requirement provided below the post, for helping the support giver to understand more about the victim.Strictly follow the question format of schema.Give only the json output as specified in the schema and no explanation needed."
    prompt="Post: "+post+"\n\n"+"event scale: "+str(row['ES'])+" effect scale: "+str(row['EFS'])+" requirement scale: "+str(row['RS'])+"\n\n"+"schema:\n"+'''{"event_question": "","effect_question": "","requirement_question": ""}'''+"\n\n"+note
    answer=f'''{{"event_question": "{str(row['EMaskingQ'].replace("X",row["EMask"]))}","effect_question": "{str(row['EFSMaskingQ'].replace("X",row["EFSMask"]))}","requirement_question": "{str(row['RMaskingQ'].replace("X",row["RMask"]))}"}}'''
    # row['text'] = f'''<|start_header_id|>system<|end_header_id|>\n\n
    #         { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
    #         {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>'''
    row['text']=prompt_template(system_prompt,prompt,answer)
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
        format_chat_template,
        num_proc=4,
    )
    val_dataset = Dataset.val_df.select(range(testsize)).map(
        format_chat_template,
        num_proc=4,
    )
    test_dataset = Dataset.test_df.select(range(testsize)).map(
        prepare_eval,
        num_proc=4,
    )
else:
    train_dataset = Dataset.train_df.map(
        format_chat_template,
        num_proc=4,
    )
    val_dataset = Dataset.val_df.map(
        format_chat_template,
        num_proc=4,
    )
    test_dataset = Dataset.test_df.map(
        prepare_eval,
        num_proc=4,
    )

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch,
    gradient_accumulation_steps=1,
    num_train_epochs=epochs,
    optim="paged_adamw_32bit",
    evaluation_strategy="epoch",
    save_strategy='epoch',  
    warmup_steps=10,
    logging_strategy="epoch",
    learning_rate=lr,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to=["wandb"],
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=context_length,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,   
    # data_collator=data_collator,
)

class CustomCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader,**kwargs):
        # print("Evaluation done")
        model.config.use_cache = True
        tokenizer.padding_side = 'left'
        val_test=val_dataset.map(prepare_eval,num_proc=4)
        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|endoftext|>")
        # ]
        dataloader = DataLoader(val_test, batch_size=eval_batch)
        preds=[]
        preds2=[]
        for batch in tqdm(dataloader,total=len(dataloader)):
            inputs = tokenizer(batch['test'], return_tensors='pt',max_length=context_length_eval,truncation=True,padding=True).to("cuda:0")
                            # max_length=context_length_eval,truncation=True
            outputs = model.generate(
                **inputs,
                # eos_token_id=terminators,
                num_return_sequences=1,
                max_new_tokens=100,
            )
            text = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            for i in text:
                preds.append(i.strip())
        # print(preds,val_test['answer'])
        x=compute_metrics(preds,val_test['answer'])
        print(x)
        wandb.log(x)
        del preds,preds2
            # return preds,val_test['answer']

if run_type=="train":
    trainer.add_callback(CustomCallback())
    # trainer.eval_dataset=eval_dataset
    val_set=copy.deepcopy(val_dataset)
    val_set=val_set.map(prepare_eval,num_proc=4)
    trainer.train()
    val_set=copy.deepcopy(test_dataset)
    trainer.evaluate()
    model.save_pretrained(new_model+"final")
    tokenizer.save_pretrained(new_model+"final")
    del model,tokenizer

elif run_type=="evaluate":
    trainer.add_callback(CustomCallback())
    val_set=copy.deepcopy(val_dataset)
    trainer.evaluate()
    val_set=copy.deepcopy(test_dataset)
    trainer.evaluate()
    del model,tokenizer
torch.cuda.empty_cache()
wandb.finish()

# val_test=val_dataset.map(prepare_eval,num_proc=4)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
# inputs = tokenizer(val_test['test'], return_tensors='pt',padding=True,max_length=4096, truncation=True).to("cuda:0")

# outputs = model.generate(
#     **inputs,
#     eos_token_id=terminators,
#     num_return_sequences=1,
# )
# # print(inputs)
# # response = [outputs[i][inputs['input_ids'].shape[-1]:] for i in range(inputs['input_ids'].shape[0])]

# text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# # print(text)
# preds=[]
# for i in text:
#     # print(i)
#     # print(i.split("assistant")[1])
#     preds.append(i.split("assistant")[1].strip())
#     # print("done")
    # # print(i.split("assistant"))
    # print(i.split("assistant")[1])


# def load_model(base_model,model_name):
#     tokenizer=AutoTokenizer.from_pretrained(model_name,truncation_side="left",padding_side="left")
#     model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     return_dict=True,
#     load_in_8bit=True,
#     device_map="cuda:0",
#     )
#     model, tokenizer = setup_chat_format(model, tokenizer)
#     model=PeftModel.from_pretrained(model,model_name)
#     return model,tokenizer

# base_model="meta-llama/Meta-Llama-3-8B-Instruct"
# new_model="llama-3-8b-instruct-2"
# model=load_model(base_model,new_model)
# def model_evaluation(model,tokenizer,val_dataset):
#     #bacth of 10 samples
    
#     val_test=val_dataset.map(prepare_eval,num_proc=4)
#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]
#     dataloader = DataLoader(val_test, batch_size=30)
#     preds=[]
#     for batch in tqdm(dataloader,total=len(dataloader)):
#         inputs = tokenizer(batch['test'], return_tensors='pt',padding=True, 
#                         max_length=2048,truncation=True).to("cuda:0")

#         outputs = model.generate(
#             **inputs,
#             eos_token_id=terminators,
#             num_return_sequences=1,
#             max_new_tokens=200,
#         )
#         text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         for i in text:
#             preds.append(i.split("assistant")[1].strip())
#             print(i.split("assistant")[1].strip())
#         del inputs,outputs,text
#         torch.cuda.empty_cache()
#     return preds,val_test['answer']

# for i in os.listdir(f"./{new_model}"):
#     print("checkpoint",i)
#     model,tokenizer=load_model(base_model,f"./{new_model}/{i}")
#     if val_set=='val':
#         preds,labels=model_evaluation(model,tokenizer,val_dataset)
#     elif val_set=='test':
#         preds,labels=model_evaluation(model,tokenizer,test_dataset)
#     elif val_set=='train':
#         preds,labels=model_evaluation(model,tokenizer,train_dataset)
#     compute_metrics(preds,labels)
#     del model,tokenizer
#     torch.cuda.empty_cache()

wandb.finish()