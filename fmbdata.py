from datasets import load_dataset

def preprocess_function(example):
    # print(example.keys())
    return {
        "prompt": [{"role": "user", "content": example["prompt"] + "\n Seed text: " + example["seed_text"]}],
        "completion": [
            {"role": "assistant", "content": f"{example['target_text']}"}  # for sft
        ],
        "ground_truth": [
            {"role": "assistant", "content": f"{example['target_text']}"}  # for grpo
        ],
    }

def init_fmb_dataset(data_path, shuffle=False, seed=42):
    # data_path = '/opt/pangu/fmbench/data_json/academic_paper.json'

    # 从 JSON 文件加载数据集
    dataset = load_dataset("json", data_files=data_path)
    dataset = dataset['train']

    col_list_to_remove = ['seed_id', 'variant_id', 'format', 'difficulty_level', 'variant_family', 'validator_spec']
    # col_list_to_remove.extend(['seed_text', 'target_text'])
    # col_list_to_remove.extend(['seed_text'])  # DO NOT REMOVE seed_text, cause error in sft
    dataset = dataset.map(preprocess_function, remove_columns=col_list_to_remove)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    return dataset

if __name__ == "__main__":
    data_path = '/opt/pangu/fmbench/data_json/academic_paper.json'
    fmb_dataset = init_fmb_dataset(data_path)
    print(fmb_dataset[0])
