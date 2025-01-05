from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import pickle
import os
import logging
from prompt_gen import PromptGen

# Setup logging
LOG_DIR = os.getenv("LOG_DIR", ".")  # Get LOG_DIR from environment or use /current_dir
LOG_FILE = os.path.join(LOG_DIR, "get_cot_batch.log")
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())  # Also log to stdout

logging.info("Starting get_cot_batch.py script...")

# Model setup
model_id = "./.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model=model_id, max_model_len=20000, tensor_parallel_size=4, gpu_memory_utilization=0.95)

def pipe_batch(batch_messages):
    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=1800)
    prompts = [
        llm.get_tokenizer().apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        for msg in batch_messages
    ]
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# Load DataFrame
logging.info("Loading data from ravis_df.pkl...")
ravis_df = pd.read_pickle("ravis_df.pkl")

# Checkpoint file
CHECKPOINT_FILE = "batch_checkpoint.pkl"

# Resume from checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    logging.info(f"Resuming from checkpoint: {CHECKPOINT_FILE}...")
    with open(CHECKPOINT_FILE, "rb") as f:
        checkpoint = pickle.load(f)
    processed_indices = checkpoint["processed_indices"]
    responses = checkpoint["responses"]
    response_dict = checkpoint.get("response_dict", {})
    ravis_df = checkpoint.get("ravis_df", ravis_df)
else:
    logging.info("No checkpoint found. Starting fresh...")
    processed_indices = set()
    responses = []
    response_dict = {}

# Process data in batches
batch_size = 1000
batch_messages = []
indices = []

try:
    logging.info("Processing data in batches...")
    for i, row in tqdm(ravis_df.iterrows(), total=len(ravis_df)):
        if i in processed_indices:
            continue  # Skip already processed rows

        encoded_related_studies = row["encoded_related_studies"]
        title = row["title"]
        description = row["description"]
        messages_instruct = [PromptGen.gen_messages(
            PromptGen.user_prompt_template(encoded_related_studies, title, description, message))
            for message in row["messages"]
        ]
        for message in messages_instruct:
            batch_messages.append(message)
            indices.append(i)
            if len(batch_messages) == batch_size:
                logging.info(f"Processing batch of size {len(batch_messages)}...")
                batch_responses = pipe_batch(batch_messages)
                responses.extend(batch_responses)
                for idx, response in zip(indices, batch_responses):
                    if idx not in response_dict:
                        response_dict[idx] = []
                    response_dict[idx].append(response)
                processed_indices.update(indices)

                # Log the last response in the batch for monitoring
                if batch_responses:
                    logging.info(f"Last response from batch: {str(batch_responses[:5])}")

                # Save checkpoint
                with open(CHECKPOINT_FILE, "wb") as f:
                    pickle.dump({
                        "processed_indices": processed_indices,
                        "responses": responses,
                        "response_dict": response_dict,
                        "ravis_df": ravis_df
                    }, f)
                logging.info(f"Checkpoint saved with {len(processed_indices)} processed rows.")
                logging.info(f"Last message from final batch: {str([str(i)[-1000:] for i in batch_messages[:5]])}")
                batch_messages = []
                indices = []

    if batch_messages:
        logging.info(f"Processing final batch of size {len(batch_messages)}...")
        batch_responses = pipe_batch(batch_messages)
        responses.extend(batch_responses)
        for idx, response in zip(indices, batch_responses):
            if idx not in response_dict:
                response_dict[idx] = []
            response_dict[idx].append(response)
        processed_indices.update(indices)

        # Log the last response in the final batch
        if batch_responses:
            logging.info(f"Last response from final batch: {str(batch_responses[:5])}")

        # Save final checkpoint
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump({
                "processed_indices": processed_indices,
                "responses": responses,
                "response_dict": response_dict,
                "ravis_df": ravis_df
            }, f)
        logging.info(f"Final checkpoint saved with {len(processed_indices)} processed rows.")

    # Map responses back to rows in the DataFrame
    ravis_df["response"] = ravis_df.index.map(lambda idx: response_dict.get(idx, []))

    # Save results
    logging.info("Saving results to response_dict.pkl and ravis_df_with_responses.pkl...")
    pickle.dump(response_dict, open("response_dict.pkl", "wb"))
    ravis_df.to_pickle("ravis_df_with_responses.pkl")

    # Remove checkpoint file after successful run
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logging.info("get_cot_batch.py script finished successfully!")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    # Save current state in case of failure
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({
            "processed_indices": processed_indices,
            "responses": responses,
            "response_dict": response_dict,
            "ravis_df": ravis_df
        }, f)
    logging.info("Checkpoint saved after failure.")
    raise  # Re-raise the exception to exit the script