# read initial retrieval results and rerank them
from pyserini.search.lucene import LuceneSearcher
from pygaggle.rerank.transformer import MonoT5
from transformers import T5ForConditionalGeneration
import json
import argparse
from tqdm import tqdm
import torch
from collections import defaultdict
from copy import deepcopy
from itertools import permutations
from typing import List

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM,
                          PreTrainedModel,
                          PreTrainedTokenizer,
                          T5ForConditionalGeneration)
from sentence_transformers import CrossEncoder
from pygaggle.rerank.base import Reranker, Query, Text
from pygaggle.rerank.similarity import SimilarityMatrixProvider
from pygaggle.model import (BatchTokenizer,
                            LongBatchEncoder,
                            QueryDocumentBatch,
                            DuoQueryDocumentBatch,
                            QueryDocumentBatchTokenizer,
                            SpecialTokensCleaner,
                            T5BatchTokenizer,
                            T5DuoBatchTokenizer,
                            greedy_decode)




def read_retrieval_results(retrieval_results_file, qrels, cut_off=100, filter_qids=False):
    '''

    :param retrieval_results_file: lines of retrieval results like: 2082 Q0 msmarco_passage_49_486599463 1 1373.633789 Anserini
    :return:
    '''
    if filter_qids:
        print('filtering queries')
    retrieval_results = {}
    with open(retrieval_results_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            qid = line[0]
            pid = line[2]
            rank = int(line[3])
            score = float(line[4])
            tag = line[5]
            if filter_qids:
                if line[0] not in qrels:
                    continue
            if qid not in retrieval_results:
                retrieval_results[qid] = {}
            if rank <= cut_off:
                retrieval_results[qid][pid] = (rank, score, tag)

    return retrieval_results

def get_text_from_index(searcher, pid):
    doc = searcher.doc(pid)
    raw_text = doc.raw()
    data = json.loads(raw_text)
    try:
        return data['passage']
    except:
        return data['contents']

def read_queries(queries_file, qrels, filter_qids=False):
    '''

    :param queries_file: format like '2000138	How does the process of digestion and metabolism of carbohydrates start'
    :return: 
    '''''
    if filter_qids:
        print('filtering queries')
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if filter_qids:
                if line[0] not in qrels:
                    continue
            qid = line[0]
            query = line[1]
            queries[qid] = query
    return queries

def read_qrels(qrels_file):
    '''

    :param qrels_file: format like '2082 0 msmarco_passage_01_552803451 0'
    :return:
    '''
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            qid = line[0]
            docno = line[2]
            qrels[qid] = docno
    return qrels

def get_formatted_query_passage(queries, retrieval_results, qid, searcher):
    '''
    

    :param queries:
    :param retrieval_results:
    :return:
    '''
    # Here's our query format:
    # query = Query('who proposed the geocentric theory')
    # Here's our text format:
    # passages = [['7744105', 'For Earth-centered it was  Geocentric Theory proposed by greeks under the guidance of Ptolemy and Sun-centered was Heliocentric theory proposed by Nicolas Copernicus in 16th century A.D. In short, Your Answers are: 1st blank - Geo-Centric Theory. 2nd blank - Heliocentric Theory.'], ['2593796', 'Copernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.he geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.'], ['6217200', 'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.opernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.'], ['3276925', 'Copernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â\x80\x93 which helped convince Galileo that the Earth was not the center of the universe â\x80\x93 can prove that ancient theory incorrect.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.'], ['6217208', 'Copernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â\x80\x93 which helped convince Galileo that the Earth was not the center of the universe â\x80\x93 can prove that ancient theory incorrect.opernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.'], ['4280557', 'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.imple tools, such as the telescope â\x80\x93 which helped convince Galileo that the Earth was not the center of the universe â\x80\x93 can prove that ancient theory incorrect. You might want to check out one article on the history of the geocentric model and one regarding the geocentric theory.'], ['264181', 'Nicolaus Copernicus (b. 1473â\x80\x93d. 1543) was the first modern author to propose a heliocentric theory of the universe. From the time that Ptolemy of Alexandria (c. 150 CE) constructed a mathematically competent version of geocentric astronomy to Copernicusâ\x80\x99s mature heliocentric version (1543), experts knew that the Ptolemaic system diverged from the geocentric concentric-sphere conception of Aristotle.'], ['4280558', 'A Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth. Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth.'], ['3276926', 'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.'], ['5183032', "After 1,400 years, Copernicus was the first to propose a theory which differed from Ptolemy's geocentric system, according to which the earth is at rest in the center with the rest of the planets revolving around it."]]
    # texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages]

    query = Query(queries[qid])
    passages = []
    for item in retrieval_results[qid]:
        pid = item
        passage = get_text_from_index(searcher, pid)
        passages.append([pid, passage])
    texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]

    return query, texts

def get_mono_t5_model(model_name_or_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model = model.to(device)
    reranker = MonoT5(model=model)
    return reranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='unicamp-dl/mt5-base-multi-msmarco', type=str, required=False,
                        help="Reranker model.")
    parser.add_argument("--initial_run", default=None, type=str, required=True,
                        help="Initial run to be reranked.")
    parser.add_argument("--corpus", default=None, type=str, required=True,
                        help="Document collection.")
    parser.add_argument("--output_run", default=None, type=str, required=True,
                        help="Path to save the reranked run.")
    parser.add_argument("--queries", default=None, type=str, required=True,
                        help="Path to the queries file.")
    parser.add_argument('--qrel_filter', default='False', type=str, required=False)
    parser.add_argument('--qrels_file', default=None, type=str, required=False,
                        help="Path to the qrels file.")
    '''
    running shell:
    python rerank.py --model_name_or_path unicamp-dl/mt5-base-multi-msmarco --initial_run runs/run.msmarco-passage.dev.small.tsv --corpus corpus/collection.tsv --output_run runs/run.msmarco-passage.dev.small.reranked --queries queries/dev.small.tsv
    '''

    args = parser.parse_args()
    # if args.corpus is a path and exists:
    import os
    if args.corpus and os.path.exists(args.corpus):
        searcher = LuceneSearcher(args.corpus)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(args.corpus)
    model = get_mono_t5_model(args.model_name_or_path)

    if args.qrel_filter != 'False':
        print(f'query filtering with qrels file: {args.qrel_filter}')
        qrels = read_qrels(args.qrels_file)
        qrel_filter = True
    else:
        print(f'no query filtering with qrels file.')
        qrels = None
        qrel_filter = False
    run = read_retrieval_results(args.initial_run, qrels, filter_qids=qrel_filter)
    queries = read_queries(args.queries, qrels, filter_qids=qrel_filter)
    # Run reranker
    trec = open(args.output_run,'w')
    # marco = open(args.output_run + '-marco.txt','w')
    for idx, query_id in enumerate(tqdm(run.keys())):
        query,texts = get_formatted_query_passage(queries, run, query_id, searcher)
        reranked = model.rerank(query, texts)
        for rank, document in enumerate(reranked):
            trec.write(f'{query_id}\tQ0\t{document.metadata["docid"]}\t{rank+1}\t{document.score}\t{args.model_name_or_path}\n')
            # marco.write(f'{query_id}\t{document.metadata["docid"]}\t{rank+1}\n')
    trec.close()
    # marco.close()
    print("Done!")
if __name__ == "__main__":
    main()