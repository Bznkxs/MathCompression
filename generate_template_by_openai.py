from openai import OpenAI
from datasets import load_dataset
import os
import os.path
import json
with open("openai_key", "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)


def generate_some_template(client, model="gpt-4o-mini", message_content=""):
    if os.path.exists("responses/index.json"):
        with open("responses/index.json", "r") as f:
            index_dict = json.load(f)
    else:
        index_dict = {}
    if message_content in index_dict:
        response_id = index_dict[message_content]
        with open(f"responses/{response_id}.json", "r") as f:
            response_dict = json.load(f)
        print("Using cached response...")
        return [response_dict["choices"][i]["message"]["content"] for i in range(len(response_dict["choices"]))], response_id
    print("Creating new response...")
    response = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message_content},
      ],
        n=10
    )
    response_dict = response.to_dict()
    os.makedirs("responses", exist_ok=True)
    with open(f"responses/{response.id}.json", "w") as f:
        json.dump(response_dict, f, indent=2)
    index_dict[message_content] = response.id
    with open("responses/index.json", "w") as f:
        json.dump(index_dict, f, indent=2)
    return [response.choices[i].message.content for i in range(len(response.choices))], response.id


def read_raw_template_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def process_raw_template(raw_template, question, answer):
    key_split_phrase = "Now consider the following:"
    part_a, part_b = raw_template.split(key_split_phrase)
    question_phrase = "Question:\n"
    part_b_0, part_b_1 = part_b.split(question_phrase)
    answer_phrase = "Answer:\n"
    part_b_1_0, part_b_1_1 = part_b_1.split(answer_phrase)

    return part_a + key_split_phrase + part_b_0 + question_phrase + question + "\n\n" + answer_phrase + "\n\n" + answer

def get_dataset(dataset_name, *args):
    ds = load_dataset(dataset_name, *args)
    return ds

def extract_template_generator_from_gpt_response(response):
    response_code = response.rsplit("```", maxsplit=2)
    if len(response_code) < 3:  # code block not found
        response_code = response.rsplit("python\n", maxsplit=1)
        if len(response_code) < 2:
            return None
    response_code = response_code[1]
    response_code = response_code.strip("python")
    response_code = response_code.strip()
    return response_code

def run_template_generator_code(template_generator_code, output_dir, times=100):
    with open("template_generator.py", "w") as f:
        f.write(template_generator_code)
    for i in range(times):
        os.system(f"python template_generator.py > {output_dir}/{i}.txt")



if __name__ == '__main__':
    ds = get_dataset("openai/gsm8k", "main")
    raw_template = read_raw_template_from_file("template.txt")
    for i in range(10):
        question = ds["train"][i]["question"]
        answer = ds["train"][i]["answer"]
        processed_template = process_raw_template(raw_template, question, answer)
        print("\n\n\n")
        print("=================================")
        print("=================================")
        print("Template:")
        print(processed_template)
        print("\n")
        print("-----------------------------")
        generated_responses, response_id = generate_some_template(client, message_content=processed_template)
        print(generated_responses[0])
        print("-----------------------------")
        for i in range(len(generated_responses)):
            generated_response = generated_responses[i]
            print(f"Response {i}:")
            template_generator_code = extract_template_generator_from_gpt_response(generated_response)
            print(template_generator_code)
            print("-----------------------------")

            output_file = f"output/output_{response_id}/template_{i}"
            os.makedirs(output_file, exist_ok=True)
            run_template_generator_code(template_generator_code, output_file)
