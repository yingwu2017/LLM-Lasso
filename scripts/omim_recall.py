from langchain_community.vectorstores import Chroma
import sys
sys.path.insert(0, ".")
import constants
from langchain_openai import OpenAIEmbeddings
import torch
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field

import os
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

@dataclass
class Arguments:
    testfile: str = field(metadata={
        "help": "File with test queries, with one query per line"
    })
    k: list[int] = field(metadata={
        "help": "number of documents to retrieve"
    })
    device: str = field(default="cpu", metadata={
        "help": "cpu or cuda device"
    })
    batch_size: int = field(default=10000)

def main(args: Arguments):
    with open(args.testfile) as f:
        test_queries = [line.strip() for line in f.readlines()]
    print(test_queries)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=constants.OMIM_PERSIST_DIRECTORY, embedding_function=embeddings)
    test_embeds = torch.Tensor(embeddings.embed_documents(test_queries)).to(args.device)

    all_docs = vectorstore._collection.get(include=[])
    all_ids = all_docs["ids"]


    ground_truth = []
    for j in range(test_embeds.shape[0]):
        test = test_embeds[j, :]
        cosine = torch.zeros(len(all_ids), device=args.device)
        for i in tqdm(range(0, len(all_ids), args.batch_size)):
            ids = all_ids[i:i+args.batch_size]
            embed = torch.from_numpy(vectorstore._collection.get(ids, include=["embeddings"])["embeddings"]).to(args.device)

            cosine[i:i+args.batch_size] = torch.sum(test * embed, dim=1) / (torch.sum(embed.square(), dim=1).sqrt() * torch.norm(test))
        argsort = torch.argsort(cosine, descending=True)
        ground_truth.append([all_ids[i] for i in argsort])

    for k in args.k:
        queried = vectorstore._collection.query(query_texts=test_queries, query_embeddings=test_embeds.tolist(), n_results=k)["ids"]
        recalls = []
        for j in range(len(test_queries)):
            recalls.append(len(set(ground_truth[j][:k]).intersection(set(queried[j]))) / k)
        print(f"Recall@{k}: {sum(recalls) / len(recalls)}")

if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    (args,) = parser.parse_args_into_dataclasses()
    main(args)

"""
python scripts/omim_recall.py --testfile prompts/recall_at_k.txt --k 1 5 10 20 50 100 1000
"""