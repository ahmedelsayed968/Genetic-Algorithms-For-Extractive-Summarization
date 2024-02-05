from datasets import load_dataset, DatasetDict
import pandas as pd
import re
from string import punctuation
from mosestokenizer import MosesDetokenizer,MosesTokenizer
def get_data(data_type:str,ds_builder:DatasetDict)->pd.DataFrame:
  df = [
            {
                "article": entry['article'],
                "summary":entry['highlights']
            }
            for entry in ds_builder[data_type]
      ]
  return df  

def load_dataset(id_name:str,config=None)->DatasetDict:
  ds = load_dataset(id_name,config)
  return ds

# consider any non english character as impurity in the data and 
RE_SUSPICIOUS = re.compile(f'{punctuation}|([^a-zA-Z])')
def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)


def yield_tokens(df:pd.DataFrame,col_name,tokenizer:MosesTokenizer):
   for _,example in df.iterrows():
      tokens = tokenizer(example[col_name])
      yield tokens
  
if __name__ == '__main__':
  pass