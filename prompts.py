def llama3_get_prompt(system_prompt, user_prompt,assistant_prompt=None):
    if assistant_prompt:
        prompt=f'''<|start_header_id|>system<|end_header_id|>\n\n
            { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_prompt}<|eot_id|>'''
    else:
        prompt=f'''<|start_header_id|>system<|end_header_id|>\n\n
                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
                { user_prompt }<|eot_id|><|start_header_id|>assistant<|end_header_id|>''' 
    return prompt

def phi3_get_prompt(system_prompt, user_prompt,assistant_prompt=None):
    if assistant_prompt:
        # prompt=f'''<|system|>
        # {system_prompt}<|end|>
        # <|user|>
        # {user_prompt}<|end|>
        # <|assistant|>
        # {assistant_prompt}<|end|>'''
        prompt=f'''
        <|user|>
        {system_prompt}\n
        {user_prompt}<|end|>
        <|assistant|>{assistant_prompt}<|end|>'''
    else:
        # prompt=f'''<|system|>
        #     {system_prompt}<|end|>
        #     <|user|>
        #     {user_prompt}<|end|>
        #     <|assistant|>'''
        prompt=f'''
        <|user|>
        {system_prompt}\n
        {user_prompt}<|end|>
        <|assistant|>'''
    return prompt


def mistral_get_prompt(system_prompt, user_prompt,assistant_prompt=None):
    pass

def gemma_get_prompt(system_prompt, user_prompt,assistant_prompt=None):
    if assistant_prompt:
        prompt=f'''<start_of_turn>user
                    {system_prompt}\n{user_prompt}<end_of_turn>
                    <start_of_turn>model
                    {assistant_prompt}<eos>'''
    else:
        prompt=f'''<start_of_turn>user
                    {system_prompt}\n{user_prompt}<end_of_turn>
                    <start_of_turn>model'''

    return prompt

