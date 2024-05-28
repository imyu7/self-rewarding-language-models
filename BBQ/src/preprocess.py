# Fair-LLM-Benchmarkから変更、self-rewarding-language-modelsで実行
# ../data/sampled_bbq.csv will be created
# if it already exists, you don't need to run this script

import pandas as pd

config = {
    "input_folder": "BBQ/data/",
    "output_train_file": "BBQ/data/sampled_bbq_train.csv", # 4 data for train
    "output_test_file": "BBQ/data/sampled_bbq_test.csv", # 100 data for test
}

# choose category
category = [
    "Age", 
    "Race_ethnicity", 
    # "Sexual_orientation", 
    # "Disability_status", 
    # "Race_x_SES", 
    "Gender_identity", 
    # "Race_x_gender", 
    # "Nationality", 
    "Religion", 
    # "Physical_appearance", 
    # "SES"
]

train_idx = {
    "Age": 40, 
    "Race_ethnicity": 42, 
    # "Sexual_orientation", 
    # "Disability_status", 
    # "Race_x_SES", 
    "Gender_identity": 41, 
    # "Race_x_gender", 
    # "Nationality", 
    "Religion": 43, 
    # "Physical_appearance", 
    # "SES"
    }

# read data
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for cat in category:
    df_cat = pd.read_json(f"{config['input_folder']}{cat}.jsonl", lines=True, orient='records')
    df_cat_test = df_cat.head(40)
    df_cat_train = df_cat.iloc[train_idx[cat], :]
    df_test = pd.concat([df_test, df_cat_test], axis=0)
    df_train = pd.concat([df_train, df_cat_train], axis=1)
df_train = df_train.T

# preprocess
def change_gender_terms(text):
    if text == "woman" or text == "girl":
        return "F"
    elif text == "man" or text == "boy":
        return "M"
    else:
        return text
    

df_train["unknown_choice"] = df_train["answer_info"].apply(lambda x: 0 if x["ans0"][1] == "unknown" else 1 if x["ans1"][1] == "unknown" else 2 if x["ans2"][1] == "unknown" else '-1')
df_train["stereotyped_choice"] = df_train.apply(
    lambda x: 
    0 if change_gender_terms(x["answer_info"]["ans0"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else 1 if change_gender_terms(x["answer_info"]["ans1"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else 2 if change_gender_terms(x["answer_info"]["ans2"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else '-1', 
    axis=1)

assert len(df_train[df_train["unknown_choice"] == '-1']) == 0
assert len(df_train[df_train["stereotyped_choice"] == '-1']) == 0


df_train.drop(columns=["answer_info", "additional_metadata"], inplace=True)
print(df_train.shape)

# save
df_train.to_csv(config["output_train_file"], index=False)

df_test["unknown_choice"] = df_test["answer_info"].apply(lambda x: 0 if x["ans0"][1] == "unknown" else 1 if x["ans1"][1] == "unknown" else 2 if x["ans2"][1] == "unknown" else '-1')
df_test["stereotyped_choice"] = df_test.apply(
    lambda x: 
    0 if change_gender_terms(x["answer_info"]["ans0"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else 1 if change_gender_terms(x["answer_info"]["ans1"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else 2 if change_gender_terms(x["answer_info"]["ans2"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else '-1', 
    axis=1)

assert len(df_test[df_test["unknown_choice"] == '-1']) == 0
assert len(df_test[df_test["stereotyped_choice"] == '-1']) == 0


df_test.drop(columns=["answer_info", "additional_metadata"], inplace=True)
print(df_test.shape)

# save
df_test.to_csv(config["output_test_file"], index=False)

