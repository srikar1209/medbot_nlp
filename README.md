
Site Url : [MedBot](https://srikar1209.github.io/medbot_nlp/)

Medical Symptom Analysis and Diagnosis:
This project aims to develop a system for analyzing medical symptoms and providing potential disease diagnoses based on a given set of symptoms. The project utilizes natural language processing techniques, machine learning, and data analysis to process medical datasets and perform symptom-disease mapping.






Installation Required:
unsloth: For loading and working with the FastLanguageModel.
trl, peft, accelerate, bitsandbytes: Packages for training and optimization.
xformers, transformers, datasets, torch: Core libraries for transformers, dataset handling, and PyTorch.




Contributing:
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
'''
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install xformers transformers datasets torch

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments
from trl import SFTTrainer
'''

# Load the dataset
'''
dataset = load_dataset('lighteval/med_dialog', 'healthcaremagic')
dataset = dataset.remove_columns(['tgt', 'id'])
'''
# Generate prompt function
```
def generate_prompt(Instruction: str, user: str, system: str) -> str:
    return f"""
    Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
```

    ### Instruction -
    {Instruction}

    ### User Input -
    {user}

    ### Your Response -
    {system}
    """

# Parse conversation function
```
def parse_conversation_to_df(text):
    text = text['src']
    data = {'prompt': ""}
    messages = text.split("Patient: ")[1:]
    instruction = """You are an AI medical assistant. Your role is to engage in a thoughtful
    dialogue with the user to fully understand symptoms and health concerns."""
    
    for msg in messages:
        msg = ' '.join(msg.strip().split())
        parts = msg.rsplit("Doctor: ", 1)
        patient_msg = parts[0].strip()
        doctor_msg = parts[1].strip() if len(parts) > 1 else ""
        patient_msg = ' '.join(patient_msg.split())
        doctor_msg = ' '.join(doctor_msg.split())
        keyword = "Regards"
        if keyword in doctor_msg:
            doctor_msg = doctor_msg.split(keyword)[0] + keyword
        data["prompt"] = generate_prompt(Instruction=instruction, system=doctor_msg.strip(), user=patient_msg.strip())
    
    return data
```
# Combine datasets
'''
combined_dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
dataset = combined_dataset.map(parse_conversation_to_df).remove_columns(['src']).with_format('pt')
'''
# Model parameters
'''
max_seq_length = 1024
model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
'''
# Apply LoRA
'''
model1 = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None,
)

# Define the trainer
trainer = SFTTrainer(
    model=model1,
    train_dataset=dataset,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=30,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# Train the model
trainer.train()
model1.save_pretrained(model_name + "_lora_model1")
'''

# Summarization function
'''
def summarize(model, tokenizer, user: str):
    instruction = """You are an AI medical assistant to have caring,
                    thoughtful dialogues to understand people's symptoms and health concerns.
                    You should provide disease name, medications needed for patient, food to avoid."""
    text = generate_prompt(Instruction=instruction, user=user, system="")
    inputs = tokenizer(text, return_tensors="pt")
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model1.generate(**inputs, max_new_tokens=220)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
    '''

# Sample user input
'''
user_input = """I am active, healthy and strong, just turned 51, female, exercise class twice a week, pretty busy, no allergies or medications.
                For the past two weeks my muscles and joints are achy and actually hurt. They feel stiff like I did a new exercise and then did not stretch.
                Have a big red bump that I thought was a black fly bite, it is sore and hard on my shin like I bumped it.
                Does not look like a tick bite. Any ideas why the aches?"""
'''

# Get summary
'''
summary = summarize(model=model1, tokenizer=tokenizer, user=user_input)
print('After Fine tuning - ', summary)
'''

# Another user input
'''
user_input2 = """I wake up every morning for the past 90 days with watery eyes and runny nose, a cough and sore throat which sometimes lasts all day.
                 What is your best suggestion? I tried several OTC medications with little relief. What can I try to help me with my condition? Thank you."""
'''
# Get another summary
'''
summary2 = summarize(model=model1, tokenizer=tokenizer, user=user_input2)
print('After Fine tuning - ', summary2)

summary = summarize(model=model1, tokenizer=tokenizer, user="""i wake up every morning for the past 90days with watery eyes and runny nose a cough and sore throat which sometimes last all day,
                                                                what is your best suggestion,
                                                                i tried several otc medication with little relief. what can i try to help me with my condition. Thank You.""")

print('After Fine tuning - ',summary)

'''


License:
This project is licensed under the MIT License.


Acknowledgments:

The medical dataset used in this project is sourced from [Source_Name].

This project utilizes the following libraries: Pandas, NumPy, NLTK, spaCy, and WordNet.

