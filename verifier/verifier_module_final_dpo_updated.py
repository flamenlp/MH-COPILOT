from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deepeval.models.base_model import DeepEvalBaseLLM
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
import torch
import re
import json
from tqdm import tqdm
import time
import traceback
from copy import deepcopy
api_key_1="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"
api_key_2="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"

class Output(BaseModel):
    event_score: int = Field(description="score for event question")
    effect_score: int = Field(description="score for effect question")
    requirement_score: int = Field(description="score for requirement question")
    reason: str = Field(description="reason for the score")
    # template_score: int = Field(description="score for actual questions completing the schema")
    # template_reason: str = Field(description="descriptive reason for the template score")

class Llama3(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
    ):
        self.model = model
        self.chat = ChatGroq(
        temperature=0.4,
        model=model,
        api_key=api_key_1
    )
        self.chat2=ChatGroq(
        temperature=0.4,
        model=model,
        api_key=api_key_2
        )
        self.chats=[self.chat,self.chat2]
        
    def load_model(self):
        return self.model

    def generate(self, prompt:str,index):
        # Set up a parser + inject instructions into the prompt template.
        # print("index",index)
        chat=self.chats[index]
        parser = JsonOutputParser(pydantic_object=Output)
        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", '''You are just judging the questions asked by a Mental Health support provider. Consider the following post by a support seeker on a Online Mental Health Platform.\n\nEvaluate the questions provided for the contextual relevance with the post and empathy for the support seeker.\n\n
                 Return a floating point score between 0 and 1 representing the contextual relevance of the question and empathy for the support seeker.  \n{format_instructions}\nNote:Strictly return a json only and nothing else.'''),
                ("human", "{prompt}"),
                ("ai", "Sure.."),
                ("human", "return the json response only"),
            ]
        )
        chain = chat_template | chat | parser
        result=chain.invoke({"prompt":prompt,"format_instructions": parser.get_format_instructions()})
        # print(result)
        return result
    async def a_generate(self, prompt: str):
        return self.generate(prompt)

    def get_model_name(self):
        return "Custom Azure OpenAI Model"



class Verifier():
    def __init__(self, model_name_cc, model_name_context,device,batch_size=1):
        self.cc = None
        self.cc_tokenizer = None
        self.device=device
        self.context_chat = None
        self.batch_size=batch_size
        self.index=0
        self.chats_len=0
        self.cc_path="./cc_saved_model"
        if(model_name_cc == "roberta"):
            self.load_roberta()

        if(model_name_context == "groq"):
            self.load_groq()
        with open("test2.json","r") as f:
            self.schema=json.load(f)

    def load_roberta(self):
        self.cc = RobertaForSequenceClassification.from_pretrained(self.cc_path,device_map=self.device)
        self.cc_tokenizer = RobertaTokenizer.from_pretrained(self.cc_path)
        self.cc2=RobertaForSequenceClassification.from_pretrained(self.cc_path,device_map=self.device)

    def load_groq(self):
        self.context_chat = Llama3("llama3-70b-8192")
        self.chats_len=len(self.context_chat.chats)
    def get_groq_score(self,post,actual,q):
        try:
            if actual[0][0]==None:
                return (0,0,0)
            x=post.split(" ")
            if len(x)>896:
                post=" ".join(post[len(x)-896:])
            else:
                post=" ".join(post)

            levels=[q['event_scale'],q['effect_scale'],q['req_scale']]
            schema={"event_question":q['event_question'],"effect_question":q['effect_question'],"requirement_question":q['requirement_question']}
                
            questions="\n".join([f"{i[0]}: {i[1]}" for i in actual])
            schema=json.dumps(schema).replace("{","").replace("}","").replace('"',"").replace("'","")
            prompt=f"Post: {post}\n questions: \n{questions}"
            if sum(levels)==0 or sum(levels)==6:
                reward=[1,1,1]
            else:
                x=self.context_chat.generate(prompt,self.index)
                x['template_score']=1
                # print(x['event_score'],x['effect_score'],x['requirement_score'])
                reward=[x['event_score']*x['template_score'],x['effect_score']*x['template_score'],x['requirement_score']*x['template_score']]
                # print(reward)
            for i in range(len(actual)):
                if levels[i]==2:
                    if actual[i][1]=="":
                        reward[i]=1
                    else:
                        reward[i]=0

                elif sum(levels)==0:
                    if i==0:
                        if "Can you tell me what" in actual[i][1]:
                            reward[i]=1
                        else:
                            reward[i]=0
                    else:
                        if actual[i][1]=="":
                            reward[i]=1
                        else:
                            reward[i]=0

                elif levels[i]==0:
                    if actual[i][1]=="":
                        reward[i]=0

                elif levels[i]==1:
                    if actual[i][1]=="":
                        reward[i]=0


            # print(schema,questions,reward)
            reward=tuple(reward)
            # print("reward",reward)
            return reward
        except Exception as e:
            print("hi exception in groq score!",e)
            time.sleep(7)
            self.index=(self.index+1)%self.chats_len
            return self.get_groq_score(post,actual,q)


    def get_groq_post_score(self,post,inputs,levels):
        q=None
        for j in self.schema:
            if str(j['event_scale'])==str(levels[0].item()) and str(j['effect_scale'])==str(levels[1].item()) and str(j['req_scale'])==str(levels[2].item()):
                q=j
                break
        return self.get_groq_score(post,inputs,q)
        # with ThreadPoolExecutor(max_workers=3) as executor:
        #     futures = [executor.submit(self.get_groq_score,post,inputs[i][1],inputs[i][0],q[l[i]]) for i in range(3)]
        #     results = [future.result() for future in as_completed(futures)]
        #     return results
    
    def get_groq_scores(self,posts,inputs,levels,verifier_batch_size):
        batch_size=verifier_batch_size
        results=[]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(self.get_groq_post_score,posts[i],inputs[i],levels[i]) for i in range(len(posts))]
            for future in tqdm(futures, total=len(futures)):
                results.append(future.result())  # Get the result of the task (not necessary for the progress bar)
                # tqdm.write(f"Completed {len(results)} of {len(futures)} tasks")
                # if len(results) % 4 == 0:
                self.index=(self.index+1)%self.chats_len 
        time.sleep(5)
            # results = [future.result() for future in as_completed(futures)]
        return torch.tensor(results)

    def prepare_cc_input(self, samples):
        inputs = []
        for sample in samples:
            input = [0,0,0]
            index = next(i for i, tup in enumerate(sample) if 'event_question' in tup[0])
            input[0] = sample[index][1]
            index = next(i for i, tup in enumerate(sample) if 'effect_question' in tup[0])
            input[1] = sample[index][1]
            index = next(i for i, tup in enumerate(sample) if 'requirement_question' in tup[0])
            input[2] = sample[index][1]
            input = self.cc_tokenizer(input,  return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            inputs.append(input)

        input_ids_tensor = torch.cat([ex['input_ids'] for ex in inputs], dim=0).to(self.device)
        attention_mask_tensor = torch.cat([ex['attention_mask'] for ex in inputs], dim=0).to(self.device)

        inputs = {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor
        }
        return inputs
    
    def check(self,sample):
        if len(sample)==3:
            if 'event_question' in sample[0][0] and "effect_question" in sample[1][0] and "requirement_question" in sample[2][0]:
                return 1
            else:
                return 0
        return -1
    
    def process_question(self,question):
        question=re.sub(r'^\s+|\s+$', '', question)
        question=question.replace('"',"")
        return question

    def prepare_input(self, samples,levels):
        batch = []
        level_pred=[]
        for i in range(len(samples)):
            # sample=samples[i].strip().replace("\n","").replace("{","").replace("}","")
            # sample=sample.split(",")
            
            # print(sample)
            try:
                sample=json.loads(samples[i])
                # sample=[(i.split(":")[0].replace('"',""),self.process_question(i.split(":")[1])) for i in sample.items()]
                sample=[(i[0],self.process_question(i[1])) for i in sample.items()]
                print(sample)
            except:
                print("invalid sample")
                sample=[("event_question",""),("effect_question",""),("requirement_question","")]
            # print(sample)
            if(self.check(sample)!=1):
                sample=[("event_question",""),("effect_question",""),("requirement_question","")]
            temp=[]
            for ques,level in zip(sample,levels[i]):
                if ques[1]=="" and level==2:
                  temp.append(1)
                else:
                    temp.append(0)
            level_pred.append(temp)
            batch.append(sample)
        return batch,torch.tensor(level_pred)
    
    def predict_cc(self, model_token_input,id):
        if id==1:
            self.cc.eval()
            with torch.no_grad():
                outputs = self.cc(**model_token_input)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = probabilities.argmax(dim=-1)
                del model_token_input
                return predicted_classes
        else:
            self.cc2.eval()
            with torch.no_grad():
                outputs = self.cc2(**model_token_input)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = probabilities.argmax(dim=-1)
                del model_token_input
                return predicted_classes
    
    def reward_calculator(self,choice1,choice2,levels,body):
        # try:
            levels=torch.stack((levels))
            levels=levels.T
            
            input_choice1,levels_pred_1=self.prepare_input(choice1,levels)
            input_choice2,levels_pred_2=self.prepare_input(choice2,levels)

            category_ip_choice1 = self.prepare_cc_input(input_choice1)
            category_ip_choice2 = self.prepare_cc_input(input_choice2)

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures=[executor.submit(self.predict_cc,category_ip_choice1,1),executor.submit(self.predict_cc,category_ip_choice2,2)]
                results = [future.result() for future in futures]

            category_res_1=results[0].reshape(-1,3).cpu()
            category_res_2=results[1].reshape(-1,3).cpu()
            # category_res_1 = self.predict_cc(category_ip_choice1).reshape(-1,3).cpu()
            # category_res_2 = self.predict_cc(category_ip_choice2).reshape(-1,3).cpu()
            
            # with ThreadPoolExecutor(max_workers=2) as executor:
            #     futures = [executor.submit(self.get_groq_scores(body, input_choice1,levels,1)),executor.submit(self.get_groq_scores(body, input_choice2,levels,1))]
            #     results = [future.result() for future in as_completed(futures)]
            # return results
            context_res_1 = self.get_groq_scores(body, input_choice1,levels,self.batch_size)
            context_res_2 = self.get_groq_scores(body, input_choice2,levels,self.batch_size)
            # context_res_1 = results[0]
            # context_res_2 = results[1]
            
            indices = torch.arange(category_res_1.size(1)).expand_as(category_res_1)
            category_res_1 = (category_res_1 == indices).int()
            category_res_2 = (category_res_2 == indices).int()


            category_res_1 = category_res_1.float()
            category_res_2 = category_res_2.float()

            category_res_1[levels_pred_1 == 1] = 1.0
            category_res_2[levels_pred_2 == 1] = 1.0

            # category_res_1=category_res_1.reshape(-1,3,1)
            # category_res_2=category_res_2.reshape(-1,3,1)
            outputs_1 = torch.mul(category_res_1,context_res_1).sum(dim=1)
            outputs_2 = torch.mul(category_res_2,context_res_2).sum(dim=1)

            # print(outputs_1,outputs_2)
            
        
        # except Exception as e:
        #     print("hi exception in rewards!",e)
        #     outputs_1=torch.full((len(choice1),), -1).float()
        #     outputs_2=torch.full((len(choice2),), -1).float()

            return outputs_1,outputs_2