import random
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import io
import time

def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

class TrainDataset(Dataset):
    def __init__(self, dataset_domain=["MP-DocVQA", "ArxivQA", "DUDE_long", "SciQAG", "SlideVQA", "TAT-DQA", "CUAD",
                                       "Wiki-ss"]):
        self.train_data = []
        self.page_image_df = None
        self.dataset_domain = dataset_domain

        for i, domain in enumerate(self.dataset_domain):
            start_time = time.time()  # Record the start time
            parquet_file = f"parquet/{domain}_filter.parquet"
            json_file = f"annotations_top1_negative/{domain}_train.jsonl"
            df = pd.read_parquet(parquet_file)
            if (i == 0):
                self.page_image_df = df
            else:
                self.page_image_df = pd.concat([self.page_image_df, df], ignore_index=False)
            sub_train_data = load_dataset("json", data_files=json_file)["train"]
            self.train_data.extend(sub_train_data)
            end_time = time.time()  # Record the end time
            time_taken = end_time - start_time
            print(f"-----reading {domain}_filter.parquet takes {time_taken} seconds-----")

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, doc_name, page_id):
        item_row = self.page_image_df[
            (self.page_image_df['file_name'] == doc_name) & (self.page_image_df['page'] == page_id)]
        if len(item_row) == 1:
            img_bytes, page_size, page_layouts = item_row["image"].iloc[0], item_row["page_size"].iloc[0], \
            item_row["layouts"].iloc[0]
            image = Image.open(io.BytesIO(img_bytes))
            return {"image": image, "page_size": page_size, "page_layouts": page_layouts, "file_name": doc_name,
                    "page_id": page_id}
        else:
            raise ValueError(f"Document {doc_name} does not have page {page_id}! Please check your data")

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        query = group['query']
        group_positives, group_negatives = group['positive_passages'], group['negative_passages']
        formated_query = format_query(query, "")
        pos_psg, neg_psg = group_positives[0], group_negatives[0]
        formated_passages = [self._get_image(pos_psg['doc_name'], pos_psg['page_id'])]
        formated_passages.append(self._get_image(neg_psg['doc_name'], neg_psg['page_id']))
        return formated_query, formated_passages

if __name__ == '__main__':
    train_dataset = TrainDataset(dataset_domain=["ArxivQA", "DUDE_long", "SciQAG", "SlideVQA", "TAT-DQA", "Wiki-ss"])
    print(f"there are {train_dataset.__len__()} isntances in the training dataset.")
    # load the query and passage of index 50 from the training dataset.
    formated_query, formated_passages = train_dataset.__getitem__(50)
    print(formated_query)
    print(formated_passages)