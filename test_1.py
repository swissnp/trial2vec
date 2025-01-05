# %% [markdown]
# 
# ## Chain of Thought (CoT) Creation
# 
# In this notebook, we aim to create a Chain of Thought (CoT) for clinical trial studies using HuggingFace's transformers and other relevant libraries. Below is a step-by-step guide to our workflow:
# 
# 1. **Setup and Initialization**:
#   - We initialize the ChromaDB client and load the SentenceTransformer model for encoding queries.
#   - We define a function to fix invalid JSON strings, which is crucial for handling the metadata of clinical trial studies.
# 
# 2. **Retrieving Relevant Studies**:
#   - We define a function `retrieve_relevant_studies` that queries the ChromaDB collection to find studies relevant to a given query, excluding the study already present in the query.
# 
# 3. **Crafting Context from Studies**:
#   - We define a function `craft_context_from_studies_documents` to create a context string from the documents of related studies. This context is used to provide examples in the CoT creation process.
# 
# 4. **Generating Messages for CoT**:
#   - We define a function `get_messages_for_create_CoT` that generates the system and user messages required for creating a CoT. These messages include the study title, description, and desired criteria.
# 
# 5. **Prompt Creation**:
#   - We define a function `get_prompt_from_studies` that uses the above functions to generate the complete prompt for a given study. This prompt includes the context from related studies and the task instructions for generating the CoT.
# 
# 6. **Model Inference**:
#   - We load the HuggingFace model and tokenizer, and define a function `pipe` to generate the CoT using the model. The function takes the generated messages as input and returns the model's output.
#   - For Gemini, can use the function in the gemini section to generate the CoT.
# 
# By following this workflow, we can systematically generate a Chain of Thought for clinical trial studies, leveraging the capabilities of HuggingFace's transformers and other relevant tools.
# ```

# %%
import faulthandler

faulthandler.dump_traceback_later(60, repeat=True)
import chromadb
from sentence_transformers import SentenceTransformer
from client import Client
from prompt_gen import PromptGen

client = chromadb.PersistentClient(path="./clinical_trials_chroma")
embed_model = SentenceTransformer("malteos/scincl")
collection = client.get_or_create_collection("clinical_trials_studies")


client = Client(
  client=client,
  collection=collection,
  embed_model=embed_model
)

prompt_gen = PromptGen(
  client=client
)


# %%


# %%
print(
    client.retrieve_relevant_studies("Effect of Kinesiotaping on Edema Management, Pain and Function on Patients With Bilateral Total Knee Arthroplasty [SEP] After being informed about the study and potential risk, all patients undergoing inpatient rehabilitation after bilateral total knee arthroplasty will have Kinesio(R)Tape applied to one randomly selected leg while the other leg serves as a control. Measurement of bilateral leg circumference, knee range of motion, numerical rating scale for pain, and selected questions from the Knee Injury and Osteoarthritis Outcome Score will occur at regular intervals throughout the rehabilitation stay. Patients will receive standard rehabilitation.", 
                              "NCT05013879"))

# %%
# from vllm import LLM, SamplingParams
# from vllm.sampling_params import GuidedDecodingParams
# from pydantic import BaseModel
# from typing import List, Optional
# from enum import Enum

# class SexEnum(str, Enum):
#     MALE = "MALE"
#     FEMALE = "FEMALE"
#     ALL = "ALL"

# class AgeGroupEnum(str, Enum):
#     CHILD = "CHILD"
#     ADULT = "ADULT"
#     OLDER_ADULT = "OLDER_ADULT"

# class Age(BaseModel):
#     Min: Optional[int] = None
#     Max: Optional[int] = None
#     AgeGroup: List[AgeGroupEnum]

# class EC(BaseModel):
#     InclusionCriteria: List[str] = []
#     ExclusionCriteria: List[str] = []

# json_schema = EC.model_json_schema()
# model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
# guided_decoding_params = GuidedDecodingParams(json=json_schema)
# sampling_params = SamplingParams(guided_decoding=guided_decoding_params, max_tokens=4096, temperature=0.2, top_k=50)
# llm = LLM(model=model_id, max_model_len=10000)
# import json
# def extract_json(output):
#         try:
#             return json.loads(output)
#         except:
#             print(output)
#             return None
# def batch_extract_ec(raw_ecs: List[str], llm):
#     base_prompt = """Using the following eligibility criteria details, generate a clean and structured JSON output that adheres to the schema provided. Do not rewrite the criteria. Just recite it in a structured JSON format."""
#     raw_ecs = [i.split("#Sex :")[0] if i is not None else "" for i in raw_ecs]
#     print(raw_ecs[0])
#     prompts = [f"{base_prompt}\n{ec}".replace("(Child: birth-17, Adult: 18-64, Older Adult: 65+)", "") for ec in raw_ecs]
#     print(prompts[0])
#     outputs = llm.generate(
#         prompts=prompts,
#         sampling_params=sampling_params
#     )
#     return [extract_json(output.outputs[0].text) for output in outputs]
    



# %% [markdown]
# ## Using Llama 3.1

# %%

# !pip show vllm
# %%
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Load the JSON data
ravis_dataset = load_dataset("ravistech/clinical-trial-llm-cancer-restructure")
## convert the dataset to a pandas DataFrame
ravis_df = pd.DataFrame(ravis_dataset['train'])
# Initialize an empty DataFrame
### =['encoded_related_studies', 'title', 'description', 'desired_criteria', 'messages'])
## add columns to the ravis_df
ravis_df["encoded_related_studies"] = None
ravis_df["title"] = None
ravis_df["description"] = None
ravis_df["desired_criteria"] = None
ravis_df["messages"] = None
ravis_df["parsed_ec"] = None
import json
import re

def parse_criteria_v8(text):
    eligibility_sections = text.split("#Eligibility Criteria:")
    parsed_results = []

    for section in eligibility_sections:
        if not section.strip():
            continue

        # Inclusion Criteria
        inclusion_match = re.search(r"Inclusion criteria\s*([\s\S]*?)(?=Exclusion criteria|##Sex)", section, re.IGNORECASE)
        inclusion_criteria = [
            line.strip() for line in inclusion_match.group(1).split('\n') 
            if line.strip() and len(line.strip()) > 3
        ] if inclusion_match else []

        # Exclusion Criteria
        exclusion_match = re.search(r"Exclusion criteria\s*([\s\S]*?)(?=##Sex)", section, re.IGNORECASE)
        exclusion_criteria = [
            line.strip() for line in exclusion_match.group(1).split('\n') 
            if line.strip() and len(line.strip()) > 3
        ] if exclusion_match else []

        # Sex
        sex_match = re.search(r"##Sex\s*:\s*(.+)", section, re.IGNORECASE)
        sex = sex_match.group(1).strip() if sex_match and sex_match.group(1).strip() else None

        # Ages
        min_age_match = re.search(r"Minimum Age\s*:\s*(\d+)\s*(\w+)", section, re.IGNORECASE)
        min_age = f"{min_age_match.group(1).strip()} {min_age_match.group(2).strip()}" if min_age_match else None

        max_age_match = re.search(r"Maximum Age\s*:\s*(\d+)\s*(\w+)", section, re.IGNORECASE)
        max_age = f"{max_age_match.group(1).strip()} {max_age_match.group(2).strip()}" if max_age_match else None

        age_group_match = re.search(r"Age Group.*:\s*(.+)", section, re.IGNORECASE)
        age_groups = [group.strip() for group in age_group_match.group(1).split(',') if group.strip()] if age_group_match else []

        # Accepts Healthy Volunteers
        healthy_volunteers_match = re.search(r"##Accepts Healthy Volunteers\s*:\s*(\w+)", section, re.IGNORECASE)
        accepts_healthy_volunteers = (
            healthy_volunteers_match.group(1).strip().lower() == "yes"
            if healthy_volunteers_match and healthy_volunteers_match.group(1).strip()
            else None
        )

        result = {
            "Inclusion Criteria": inclusion_criteria,
            "Exclusion Criteria": exclusion_criteria,
            "Sex": sex,
            "Minimum Age": min_age,
            "Maximum Age": max_age,
            "Age Groups": age_groups,
            "Accepts Healthy Volunteers": accepts_healthy_volunteers
        }

        parsed_results.append(json.dumps(result))  # Convert to JSON string

    return parsed_results

for i in tqdm(range(len(ravis_df))):
    ravis_df.loc[i, "parsed_ec"] = parse_criteria_v8(ravis_df.loc[i, "criteria"])
ravis_df

## drop the row if the parsed_ec Inclusion Criteria and Exclusion Criteria are empty
ravis_df = ravis_df[ravis_df["parsed_ec"].apply(lambda x: any(json.loads(x[0])["Inclusion Criteria"]) or any(json.loads(x[0])["Exclusion Criteria"]))]

## drop the row if the parsed_ec Inclusion Criteria and Exclusion Criteria are empty and more than 1
ravis_df = ravis_df[ravis_df["parsed_ec"].apply(lambda x: len(x) == 1)]



# %%
## save the ravis_df to a temp pickle file
# ravis_df.to_pickle("ravis_df.pkl")
# load the ravis_df from the pickle file
# import pandas as pd
# ravis_df = pd.read_pickle("ravis_df.pk")


# %%


# %%
## re-run the batch_extract_ec function on the unparsed_df
# from tqdm import tqdm

# batch_size = 100
# for i in tqdm(range(0, len(unparsed_df), batch_size)):
#     raw_ecs = unparsed_df.iloc[i:i+batch_size]["criteria"].tolist()
#     print(raw_ecs)
#     parsed_ecs = batch_extract_ec(raw_ecs, llm)
#     print(parsed_ecs)
#     unparsed_df.loc[i:i+batch_size-1, "parsed_ec"] = parsed_ecs


# %%
from tqdm import tqdm
def get_demographic_criteria_message(sex, min_age, max_age, age_groups):
    message = ""
    if sex:
        message += f"Sex: {sex} "
    if min_age:
        message += f"with a minimum age of {min_age} "
    if max_age:
        message += f"and a maximum age of {max_age} "
    if age_groups:
        message += f"and the following age groups (Child: birth-17, Adult: 18-64, Older Adult: 65+): {', '.join(age_groups)} "
    return message


for i, row in tqdm(ravis_df.iterrows(), total=len(ravis_df)):
    # print(f"Processing {study}")
    ## select metadata	data	criteria
    study = row[["metadata", "data", "parsed_ec"]]
    info_for_prompt = prompt_gen.get_info_for_prompt_gen(study)
    if info_for_prompt:
        encoded_related_studies, title, description, desired_criteria = info_for_prompt
        inclusion_criteria = desired_criteria.get("Inclusion Criteria", [])
        exclusion_criteria = desired_criteria.get("Exclusion Criteria", []) 
        sex = desired_criteria.get("Sex", None)
        min_age = desired_criteria.get("Minimum Age", None)
        max_age = desired_criteria.get("Maximum Age", None)
        age_groups = desired_criteria.get("Age Groups", [])
        accepts_healthy_volunteers = desired_criteria.get("Accepts Healthy Volunteers", None)
        inclusion_criteria_messages = [
            f"Inclusion Criteria: {criteria}" for criteria in inclusion_criteria
        ]
        exclusion_criteria_messages = [
            f"Exclusion Criteria: {criteria}" for criteria in exclusion_criteria
        ]
        get_demographic_criteria_messages = [
            get_demographic_criteria_message(sex, min_age, max_age, age_groups)
        ]
        ## only accepts_healthy_volunteers true or false strict
        accepts_healthy_volunteers_message = []
        if accepts_healthy_volunteers is not None:
            if accepts_healthy_volunteers == True:
                message = "Accepts Healthy Volunteers: Yes"
            else:
                message = "Accepts Healthy Volunteers: No"
            accepts_healthy_volunteers_message = [message]
        messages = inclusion_criteria_messages + exclusion_criteria_messages + get_demographic_criteria_messages + accepts_healthy_volunteers_message
        
        # Add the data to the DataFrame exsisting ravis_df
        ravis_df.at[i, "encoded_related_studies"] = encoded_related_studies
        ravis_df.at[i, "title"] = title
        ravis_df.at[i, "description"] = description
        ravis_df.at[i, "messages"] = messages
        
        if i % 1000 == 0:
            print(f"Prompt: {messages}")
        
## save the ravis_df to a temp pickle file
ravis_df.drop(columns=["__index_level_0__", "desired_criteria", "data"], inplace=True)
ravis_df.to_pickle("ravis_df.pkl")

# %%

def pipe_batch(batch_messages):
    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=4096)
    prompts = [
        tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        for msg in batch_messages
    ]
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

batch_size = 200  # Batch size for processing
responses = []

# Iterate through DataFrame and process in batches
batch_messages = []
indices = []

for i, row in tqdm(ravis_df.iterrows(), total=len(ravis_df)):
    encoded_related_studies = row["encoded_related_studies"]
    title = row["title"]
    description = row["description"]
    messages_instruct = [PromptGen.gen_messages(PromptGen.get_messages_for_CoT_huggingface(encoded_related_studies, title, description, message)) for message in row["messages"]]
    for message in messages_instruct:
        batch_messages.append(message)
        indices.append(i)
        # If batch is full, process it
        if len(batch_messages) == batch_size:
            batch_responses = pipe_batch([(i) for i in batch_messages])
            responses.extend(batch_responses)
            batch_messages = []
            indices = []

# Process remaining messages in the last batch
if batch_messages:
    batch_responses = pipe_batch(batch_messages)
    responses.extend(batch_responses)

# Map responses back to rows in the DataFrame
response_dict = {}
for idx, response in zip(indices, responses):
    if idx not in response_dict:
        response_dict[idx] = []
    response_dict[idx].append(response)
ravis_df["response"] = None
ravis_df["response"] = ravis_df.index.map(lambda idx: response_dict.get(idx, []))

# Save the updated DataFrame if needed
ravis_df.to_pickle("ravis_df_with_responses.pkl")

# %%


# %% [markdown]
# ## Using Gemini

# %%
# import time
# import pandas as pd
# import json
# from datasets import load_dataset
# from tqdm import tqdm
# import vertexai
# from vertexai.batch_prediction import BatchPredictionJob
# import os
# import uuid

# PROJECT_ID = "PROJECT-ID"  # update with Google Cloud project ID
# BUCKET_NAME = "BUCKET-NAME" # create a bucket in Google Cloud Storage first
# OUTPUT_DIR = "./output/"
# CSV_OUTPUT_PATH = "./responses_gemini.csv"
# vertexai.init(project=PROJECT_ID, location="us-central1")

# ravis_dataset = load_dataset("ravistech/clinical-trial-llm-cancer-restructure")

# batch_prompts = []
# info_data = {}  # key is uuid, value is the info for the prompt
# for study in tqdm(ravis_dataset['train']):
#     info_for_prompt = prompt_gen.get_info_for_prompt_gen(study)
    
#     if info_for_prompt:
#         unique_id = str(uuid.uuid4())  # gen uuid for each entries
#         encoded_related_studies, title, description, desired_criteria = info_for_prompt
#         messages = prompt_gen.user_prompt_template(encoded_related_studies, title, description, desired_criteria)

#         request_format = {
#             "id": unique_id,
#             "request": {
#                 "contents": [
#                     {
#                         "role": "user",
#                         "parts": [
#                             {"text": messages}
#                         ]
#                     }
#                 ],
#                 "system_instruction": {
#                     "parts": [
#                         {"text": prompt_gen.system_prompt}
#                     ]
#                 }
#             }
#         }
#         batch_prompts.append(request_format)
        
#         # store info with unique ID for later matching with the response
#         info_data[unique_id] = {
#             "encoded_related_studies": encoded_related_studies,
#             "title": title,
#             "description": description,
#             "desired_criteria": desired_criteria,
#             "messages": messages,
#             "response": None
#         }

# input_jsonl_path = f'gs://{BUCKET_NAME}/prompt_for_batch_gemini_predict.jsonl'
# with open('./prompt_for_batch_gemini_predict.jsonl', 'w') as f:
#     for prompt in batch_prompts:
#         f.write(json.dumps(prompt) + '\n')

# # Upload the JSONL file to GCS
# !gsutil cp ./prompt_for_batch_gemini_predict.jsonl {input_jsonl_path}

# batch_prediction_job = BatchPredictionJob.submit(
#     source_model="gemini-1.5-flash-002",
#     input_dataset=input_jsonl_path,
#     output_uri_prefix=f'gs://{BUCKET_NAME}/output/',
# )

# print(f"Job resource name: {batch_prediction_job.resource_name}")
# print(f"Model resource name with the job: {batch_prediction_job.model_name}")
# print(f"Job state: {batch_prediction_job.state.name}")

# # Check the job status
# while not batch_prediction_job.has_ended:
#     time.sleep(5)
#     batch_prediction_job.refresh()

# # Process the output if the job is successful
# if batch_prediction_job.has_succeeded:
#     print("Job succeeded!")
#     output_location = batch_prediction_job.output_location + "/predictions.jsonl"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     local_output_path = os.path.join(OUTPUT_DIR, "predictions.jsonl")

#     # Download the output file from GCS
#     !gsutil cp {output_location} {local_output_path}
#     print(f"Output file downloaded to: {local_output_path}")
    
#     with open(local_output_path, 'r') as f:
#         for line in f:
#             response_data = json.loads(line)
#             unique_id = response_data.get("id")
#             response = response_data.get("response", {})
            
#             if unique_id in info_data:
#                 info_data[unique_id]["response"] = response

#     df = pd.DataFrame.from_dict(info_data, orient="index")
#     df.to_csv(CSV_OUTPUT_PATH, index=False)
#     print(f"Responses saved to CSV at: {CSV_OUTPUT_PATH}")
# else:
#     print(f"Job failed: {batch_prediction_job.error}")

# print(f"Job output location: {batch_prediction_job.output_location}")




