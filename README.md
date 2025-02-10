# Fine_tuning_DeepSeek_R1

Python libraries and frameworks
The Python libraries and frameworks that will be required for fine-tuning LLMs are, -

unsloth, this package makes finetuning large language models like Llama-3, Mistral, Phi-4 and Gemma 2x faster, uses 70% less memory, and with no degradation in accuracy! You can read more here.
torch, this package is the fundamental building block for deep learning with PyTorch. It provides a powerful tensor library, similar to NumPy, but with the added advantage of GPU acceleration, which is a crucial thing when it comes to working with LLMs.
transformers is a powerful and popular open-source library for natural language processing (NLP). It provides easy-to-use interfaces for a wide range of state-of-the-art pre-trained models. As pre-trained models form the base of any fine-tuning task, this package helps in easily accessing trained models.
The trl package in Python is a specialized library for Reinforcement Learning (RL) with transformer models. It's built on top of the Hugging Face transformers library, leveraging its strengths to make RL with transformers more accessible and efficient.
Computational Requirements
Fine-tuning a model is a technique to make the LLM’s response more structured and domain-specific. There are many techniques that are adopted to fine-tune a model and some facilitate the process without actually performing a full parameter training.

However, the process of fine-tuning bigger LLMs is still not feasible for most of the average computer hardware, as all the trainable parameters along with the actual LLM are stored in the vRAM (Virtual RAM) of the GPU, and the huge size of LLMs pose a major obstacle in achieving that.

So, for the sake of this article, we will be fine-tuning the distilled version of DeepSeek-R1 LLM, which is DeepSeek-R1-Distill with 4.74 billion parameters. This LLM requires at least 8–12 GB of vRAM, and to make it accessible to all the people, we will be using Google Colab’s free T4 GPU, which has 16 GB of vRAM.

Data preparation strategies
For fine-tuning an LLM, we need structured and task-specific data. There are many data preparation strategies, be it scrapping social media platforms, websites, books or research papers.

For this article, we will be using the datasets library to load the data present in the Hugging Face Hub. We will be using HumanLLMs/Human-Like-DPO-Dataset dataset from Hugging Face, you can explore the dataset here.

Python Implementation
Installing the packages
One major benefit of using Google Colab for this fine-tuning task is that most of the packages come already installed. And we are just required to install one package, i.e. unsloth.

The process to install the package is, —

!pip install unsloth
Initializing the model and tokenizer
We will be using the unsloth package to load the pre-trained model, because it offers many useful techniques that will help us in faster downloading and fine-tuning of LLM.

The code for loading the model and tokenizer is, —

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
Here we have specified the model name, 'unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit', which is to access the pre-trained DeepSeek-R1-Distill model.
We have defined the max_seq_length to 2048, which sets the maximum length of the input sequence that the model can process. By setting it reasonably, we can optimize memory usage and processing speed.
dtype is set to None, which facilitates the mapping of the data type in which the model will be fetched, compatible with the hardware available. By using this, we don't have to explicitly check and mention the data type, unsloth takes care of all.
load_in_4bit enhances inference and reduces memory usage. Basically, we are quantizing the model to 4bit precision.
Adding LoRA Adapters
We will be adding LoRA matrices to the pre-trained LLM, which will help in fine-tuning the responses of the model. And using unsloth, the whole process is just a few lines away.

Here’s how it’s done,-

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Can be set to any, but = 0 is optimized
    bias = "none",    # Can be set to any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3927,
    use_rslora = False,  # unsloth also supports rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
Explanation of the code, —

Now, we have re-initialized the model using the get_peft_model from the FastLanguageModel, for using the PEFT technique.
We are also required to pass the pre-trained model, which we fetched in the previous step.
Here, the r=64 parameter defines the rank of the low-rank matrices in LoRA adaptation. This rank generally yields the best results when in the range of 8–128.
The lora_dropout parameter introduces dropout to the low-rank matrices during the training of this LoRA adapter model. This parameter prevents the model from being overfitted. unsloth provides us the facility to automatically choose the optimized value by setting it to 0.
The target_modules specifies the list of names of the specific classes or modules within the model that we want to apply to the LoRA adaptation.
Data Preparation
Now, that we have set the LoRA adapters on the pre-trained LLM, we can move towards structuring the data that will be used for training the model.

To structure the data, we have to specify the prompt in such a way that contains the instructions and response.

Instructions, signify the main query to the LLM. This is the question that we have asked from the LLM.
Response, signify the output from the LLM. It is used to mention how the response from the LLM should be tailored to the specific instruction(query).
The structure of the prompt is, —

human_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
We have created a function that will properly structure all the data in human_prompt, which is, —

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_human_prompts_func(examples):
    instructions = examples["prompt"]
    outputs      = examples["chosen"]
    texts = []

    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = human_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
Now, we have to load the dataset that will be used for fine-tuning the model, in our case it’s the “HumanLLMs/Human-Like-DPO-Dataset” from the Hugging Face Hub. You can explore the dataset here.

from datasets import load_dataset
dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split = "train")
dataset = dataset.map(formatting_human_prompts_func, batched = True,)
Training the Model
Now that we have both the structured data and model with LoRA adapters or matrices, we can proceed towards the training of the model.

To train the model, we have to initialize certain hyperparameters, which would facilitate the training process and will also affect the accuracy of the model to a certain aspect.

We will initialize a trainer using SFTTrainer and the hyperparameter.

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model, # The model with LoRA adapters
    tokenizer = tokenizer, # The tokenizer of the model
    train_dataset = dataset, # The dataset to use for training
    dataset_text_field = "text", # The field in the dataset that contains the structured data
    max_seq_length = 2048, # Max length of input sequence that the model can process
    dataset_num_proc = 2, # Noe of processes to use for loading and processing the data
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Batch size per GPU
        gradient_accumulation_steps = 4, # Step size of gradient accumulation
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 120, # Maximum steps of training
        learning_rate = 2e-4, # Initial learning rate
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # The optimizer that will be used for updating the weights
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
Now to start the training of the model, using this trainer, —

trainer_stats = trainer.train()
This would start the training of the model and will log all the steps with their respective Training Loss on the kernel.


Screenshot of training the model on Google Colab
Inferencing the Fine-Tuned Model
Now, as we have completed the training of the model, all we have to do is infer the fine-tuned model to evaluate its responses.

The code to do inferencing on the model is, —

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    human_prompt.format(
        "Oh, I just saw the best meme - have you seen it?", # instruction
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)
tokenizer.batch_decode(outputs)
Explanation of the code, —

We have used FastLanguageModel from the unsloth package to load the fine-tuned model for inferencing. This method yields faster results.
In order to infer the model, we have to first convert the query into the structured prompt and then tokenize the prompt.
We have also set return_tensors="pt" in order to make the tokenizer return a PyTorch tensor and then we loaded that tensor to the GPU using .to("cuda"), to increase the speed of processing.
Then we called model.generate() to generate the response for the query.
While generating we have mentioned max_new_tokens=1024, which mentions the maximum number of tokens that the model can generate.
use_cache=True helps in speeding up the generation, especially for longer sequences.
Then at last we decoded the output from the fine-tuned model from tensor to text.

Screenshot of Output from the Fine-Tuned Model
Results from the Fine-Tuned Model
This section of the article contains some other results from the fine-tuned model.

Query — 1: I love reading and writing, what are your hobbies?


Output corresponding to Query — 1
Query — 2: What’s your favourite type of cuisine to cook or eat?


Output corresponding to Query — 2
Here, one can note the level of expressiveness in the response that has been generated. The responses are more intriguing while maintaining the actual quality of the response.

Saving the Fine-Tuned Model
This step completes the whole process of fine-tuning the model, and now we can save the fine-tuned model for inferencing or using it in future.

We are also required to save the tokenizer with the model. The following is the way by which one can save their fine-tuned model on the Hugging Face Hub.

# Pushing with 4bit precision
model.push_to_hub_merged("<YOUR_HF_ID>/<MODEL_NAME>", tokenizer, save_method = "merged_4bit", token = "<YOUR_HF_TOKEN>")

# Pushing with 16 bit precision 
model.push_to_hub_merged("<YOUR_HF_ID>/<MODEL_NAME>", tokenizer, save_method = "merged_16bit", token = "<YOUR_HF_TOKEN>")
Here, you have to set the name of the model, which will be used for setting the ID of the model on the Hugging Face Hub.
One can upload the complete merged model with either 4bit or 16bit precision. The merged model signifies that the pre-trained model along with the LoRA matrices is uploaded on the hub, whereas there are options that one can only push the LoRA matrices except for the model.
You can get your Hugging Face token here.
You can find the 16bit precision model, I fine-tuned along with this article here.

Conclusion
Yup! That’s it for this article. Here are the major topics discussed in this article, —

Fine Tuning is a process by which we can make Large Language Models respond in a way, we want. Especially done to make the responses domain-specific and well-structured.
We defined a structure for arranging the dataset in the way, used for fine-tuning the model.
The main Python libraries and frameworks that we used are unsloth, torch, transformers, and trl. Along with that, we discussed about the computational requirements for fine-tuning an LLM.
We discussed a number of hyper-parameters that affect the quality of the responses generated from the fine-tuned model and also tried to initialise them for achieving our specific usecase.
We also merged the LoRA adapters or matrices with the pre-trained model for pushing it to the Hugging Face Hub.
