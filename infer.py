import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DataParallel

def do_inference(prompt):
    # Check for GPU availability and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()  # Get the number of available GPUs
    else:
        device = torch.device("cpu")
        n_gpus = 1

    tokenizer = AutoTokenizer.from_pretrained(
        "LLM360/CrystalCoder",
        revision="CrystalCoder_phase1_checkpoint_055500",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "LLM360/CrystalCoder",
        revision="CrystalCoder_phase1_checkpoint_055500",
        trust_remote_code=True
    )

    # If multiple GPUs are available, wrap the model with DataParallel
    if n_gpus > 1:
        model = DataParallel(model)

    model = model.to(device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    # Unwrap the model from DataParallel if necessary
    if isinstance(model, DataParallel):
        model = model.module

    gen_tokens = model.generate(input_ids, do_sample=True, max_length=400, pad_token_id=tokenizer.eos_token_id)

    print(tokenizer.batch_decode(gen_tokens)[0])

# Open the file in read mode
with open('read.tex', 'r') as file:
    # Read the contents of the file
    prompt = file.read()

do_inference(prompt)