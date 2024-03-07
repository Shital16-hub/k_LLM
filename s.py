import json
import csv
from infer import *
import csv
import pandas as pd

def generate_prompts(csv_file):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Iterate through each row
        for index, row in df.iterrows():
            # Extract the question from the current row
            question_ans = row['question']
            # Create the prompt
            prompt = f"Question: {question_ans}\nAnswer: \n\nSpecify only A , B, C or D if the answer is from the options and give in format 'Answer- '"
            # Yield the prompt along with the question
            yield (prompt, question_ans)
                
    except Exception as e:
        yield f"Error: {e}"

# # Example usage:
# csv_file = 'D:\My Work\Generative AI\LLM\LLM 360\ARC-Challenge-Dev.csv'
# for prompt in generate_prompts(csv_file):
#     print(prompt)





def create_csv(json_file):
    result_file = f"result_{json_file.split('.')[0]}.csv"
    
    with open(result_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Question', 'Inference Result'])
        
        for prompt, question_ans in generate_prompts(csv_file):
              inference_result = do_inference(prompt)
              csv_writer.writerow([question_ans, inference_result])


csv_files = ["ARC-Easy-Dev.csv"]

for csv_file in csv_files:
    create_csv(csv_file)
