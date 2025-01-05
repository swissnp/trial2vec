# %%
# import pandas as pd
# # Load the CSV file
# df = pd.read_csv('responses_gemini_cleaned_2024-12-13_22-19-59.csv')

# # Display the head of the dataframe
# df

# %%
# ## Add a new column to the dataframe uuid to store the unique identifier
# import uuid
# df['uuid'] = [uuid.uuid4() for _ in range(len(df.index))]
# df

# augmented_df = df[["uuid","encoded_related_studies",	"title",	"description",	"desired_criteria",	"messages"	]]
# augmented_df


# %%
# def gen_shitty_criteria_prompt_wo_retrive_and_cot(title: str, description: str):
#       return f"""
# Target Study Title: {title}
# Target Study Description: {description}

# Task Instruction:
# 1. Based on the "Target Study Title" and "Target Study Description" of the target study, please create a Eligibility Criteria for the target study.
# 4. Please provide the Eligibility Criteria in the following format (the item within the square brackets [] are the options that you can choose from):
# <FORMAT>
# #Eligibility Criteria:
# Inclusion Criteria:

# * Inclusion Criteria 1
# * Inclusion Criteria 2
# * Inclusion Criteria 3
# * ...

# Exclusion Criteria:

# * Exclusion Criteria 1
# * Exclusion Criteria 2
# * Exclusion Criteria 3
# * ...

# ##Sex :
# [MALE|FEMALE|ALL]
# ##Ages : 
# - Minimum Age : ... Years
# - Maximum Age : ... Years
# - Age Group (Child: birth-17, Adult: 18-64, Older Adult: 65+) : [ADULT|CHILD|OLDER ADULT] comma separated

# ##Accepts Healthy Volunteers:
#  [YES|NO]
# </FORMAT>
# <Example>
# #Eligibility Criteria:
# Inclusion Criteria:

# * Ability to give informed consent.
# * Patients with primary or recurrent papillary Non muscle invasive bladder cancer (NMIBC).
# * Complete transurethral resection of bladder tumor(TURBT).
# * Normal cardiac, hematological, and renal functions.
# * Patients with intermediate and high risk NMIBC confirmed by histopathology.

# Exclusion Criteria:

# * Inability to give informed consent.
# * Patients with history of previous radiotherapy or systemic chemotherapy.
# * Patients suffering from immuno-deficiency or other malignancies.
# * Patients with history of hypersensitivity reaction to epirubicin.
# * Examination under anesthesia (EUA) reveals palpable bladder mass.
# * Patients with primary, single, less than 1cm papillary bladder tumor (high likelihood of being low risk).
# * Suspicion of perforation of the bladder during TURBT.
# * Patients who develop hematuria in the recovery room necessitating continuous bladder wash or endoscopic haemostasis.
# * Patients with proven low risk NMIBC on histopathology.
# ##Sex :
# ALL
# ##Ages : 
# - Minimum Age : 18 Years
# - Age Group (Child: birth-17, Adult: 18-64, Older Adult: 65+) : ADULT, OLDER_ADULT

# ##Accepts Healthy Volunteers:
#  No
# </Example>
# """

# %%

# from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
number_gpus = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=20000)

def pipe(messages):
    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=4096)
    prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    outputs = llm.generate(prompts, sampling_params)
    return [i.outputs[0].text for i in outputs]


# %%
# def get_messages(prompt):
#   return [
#       {"role": "system", "content": "You are a eligibility criteria chatbot who always responds eligibility criteria in the given format!"},
#       {"role": "user", "content": prompt},
#   ]
# augmented_df["shitty_criteria_prompt_wo_retrive_and_cot"] = ""
# ## generate the prompts and batch the messages into pipes of 20 also save the outputs to the dataframe
# batch_size = 20
# for i in tqdm(range(0, len(augmented_df), batch_size)):
#     batch = augmented_df.iloc[i:i+batch_size]
#     messages = [get_messages(gen_shitty_criteria_prompt_wo_retrive_and_cot(title, description)) for title, description in zip(batch["title"], batch["description"])]
#     outputs = pipe(messages)
#     print(outputs[-1])
#     print(len(augmented_df.loc[i:i+batch_size-1, "shitty_criteria_prompt_wo_retrive_and_cot"]))
#     augmented_df.loc[i:i+batch_size-1, "shitty_criteria_prompt_wo_retrive_and_cot"] = outputs
# augmented_df





# %%


# %%
import pickle
import pandas as pd

# load  augmented_df.pkl from disk 
augmented_df = pd.read_pickle("augmented_df.pkl")
augmented_df

# %%
from prompt_gen import PromptGen

augmented_df["shitty_criteria_CoT"] = ""
## generate the prompts and batch the messages into pipes of 20 also save the outputs to the dataframe
batch_size = 20
from tqdm import tqdm
for i in tqdm(range(0, len(augmented_df), batch_size)):
    batch = augmented_df.iloc[i:i+batch_size]
    messages = [PromptGen.gen_messages(PromptGen.user_prompt_template(encoded_related_studies, title, description, shitty_criteria)) for encoded_related_studies, title, description, shitty_criteria in zip(batch["encoded_related_studies"],batch["title"], batch["description"], batch["shitty_criteria_prompt_wo_retrive_and_cot"])]
    outputs = pipe(messages)
    ## print outputs with the prompt
    for j, (output, message) in enumerate(zip(outputs, messages)):
        print(f"Prompt: {message[1]['content']}")
        print(f"Output: {output}")
        print("\n")
    print(len(augmented_df.loc[i:i+batch_size-1, "shitty_criteria_CoT"]))
    augmented_df.loc[i:i+batch_size-1, "shitty_criteria_CoT"] = outputs
augmented_df

augmented_df.to_pickle("augmented_df_with_CoT.pkl")


