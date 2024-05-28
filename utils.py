import os
from dotenv import load_dotenv
import re
import transformers
import torch

load_dotenv(".env")

def prepare_model(model_name):
    if model_name in ['gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', 'gpt-4-turbo-2024-04-09', "davinci-002", "gpt-3.5-turbo-instruct-0914", "babbage-002"]:
        from openai import OpenAI
        global client 
        client = OpenAI()
    elif model_name in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-v0.1"]:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        global tokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        pipeline = transformers.pipeline(
            'text-generation', 
            model=model_name, 
            torch_dtype=torch.float16,
            device_map="auto")

def get_completion(model_name, system_prompt, user_prompt, args):
    if model_name in ['gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', 'gpt-4-turbo-2024-04-09']:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **args,
        )
        # print("----------------------")
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    elif model_name in ["davinci-002", "gpt-3.5-turbo-instruct-0914", "babbage-002"]:
        response = client.completions.create(
            model=model_name,
            # prompt=f"{system_prompt}\n{user_prompt}",
            prompt=user_prompt,
            **args,
        )
        return response.choices[0].text
    elif model_name in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-v0.1"]:
        sequences = pipeline(
            user_prompt,
            # f"{system_prompt}\n{user_prompt}",
            do_sample=False, # これで再現性OK?
            max_new_tokens=args["max_tokens"],
            # temperature=args["temperature"]+0.01,
            eos_token_id=tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"]


# 使わない
# def exact_match(prediction, answer):
#     prediction = prediction.lower()
#     prediction = re.sub(r'\.', '', prediction)
#     answer = str(answer).lower()
#     return prediction == answer

# 
def include_match(prediction, answer):
    prediction = prediction.lower()
    prediction = re.sub(r'\.', '', prediction)
    prediction = prediction[-30:]
    answer = str(answer).lower()
    return answer in prediction

def inference_all(model_name, df, system_prompt, user_prompt, context_columns, answer_column, args, check_answer=True):
    prepare_model(model_name)
    for index, row in df.iterrows():
        data = {col: row[col] for col in context_columns}
        formatted_user_prompt = user_prompt.format(**data)
        # print(formatted_user_prompt)
        df.at[index, "Prediction"] = get_completion(model_name, system_prompt, formatted_user_prompt, args)
    if check_answer:
        # df["ExactMatch"] = df.apply(lambda row: exact_match(row["Prediction"], row[answer_column]), axis=1)
        df["IncludeMatch"] = df.apply(lambda row: include_match(row["Prediction"], row[answer_column]), axis=1)
    return df

def get_accuracy_all(df, exact_match=False):
    if exact_match:
        return df["ExactMatch"].mean()
    else:
        return df["IncludeMatch"].mean()
    

def get_group_accuracy(df, group_column, exact_match=False):
    group_accuracy = df.groupby(group_column).apply(lambda x: get_accuracy_all(x, exact_match))
    return group_accuracy