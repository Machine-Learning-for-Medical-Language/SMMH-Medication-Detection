import re
from locale import Error

from nltk.tokenize import TweetTokenizer

import read_files as read

###### field = [tweet_id	user_id	created_at	text	start	end	span	drug]
######   empty value "-"

# mapping_rules = []


def add_tweet_info(tweet_dict, info):
    tweet_id, user_id, created_at, text, start, end, span, drug = info[:8]
    if tweet_id not in tweet_dict:
        tweet_dict[tweet_id] = {
            "user_id": user_id,
            "created_at": created_at,
            "text": text,
            "offsets": [(start, end)],
            "span": [span],
            "drug": [drug]
        }
    else:
        tweet_dict[tweet_id]["offsets"].append((start, end))
        tweet_dict[tweet_id]["span"].append(span)
        tweet_dict[tweet_id]["drug"].append(drug)
    return tweet_dict


def get_spans(tokens, txt):
    span_info = []
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        span_info.append((token, offset, offset + len(token)))
        offset += len(token)
    return span_info


def io_tagging_process(tweet_dict):
    tweet_io_inputs = []
    i = 0
    tweet_tokenizer = TweetTokenizer()
    tokenizer = re.compile(r"\d+|[^\d\W]+|\S")
    missing = []
    for tweet_id, tweet_info in tweet_dict.items():
        tweet_text = tweet_info["text"]
        offsets = tweet_info["offsets"]
        medication_text = tweet_info["span"]
        offset_index = []
        if "-" not in offsets[0][0]:
            for item in offsets:
                offset_index += [
                    i for i in range(int(item[0]),
                                     int(item[1]) + 1)
                ]
        tweet_tokens_tokenized = tweet_tokenizer.tokenize(tweet_text)
        tweet_tokens_tokenized_1 = []
        for token in tweet_tokens_tokenized:
            if "http" not in token and "@" not in token:
                token = [m.group() for m in tokenizer.finditer(token)]
                tweet_tokens_tokenized_1 += token
            else:
                tweet_tokens_tokenized_1.append(token)

        token_span_info = get_spans(tweet_tokens_tokenized_1, tweet_text)

        token_span_info_processed = []

        for item in token_span_info:
            if "http" in item[0]:
                token_span_info_processed.append(("URL", item[1], item[2]))
            elif item[0][0] == "@":
                token_span_info_processed.append(("@USER", item[1], item[2]))
            else:
                token_span_info_processed.append((item[0], item[1], item[2]))
        tweet_input = [item[0] for item in token_span_info_processed]

        tweet_label = []
        for item in token_span_info_processed:
            if any([i in offset_index for i in range(item[1], item[2])]):
                tweet_label.append("I-Medication")
            else:
                tweet_label.append("O")

        if "I-Medication" not in tweet_label:
            # raise Error("issues with tokenizations")
            i += 1
            missing.append([medication_text, tweet_text])

        tweet_io_inputs.append([" ".join(tweet_label), " ".join(tweet_input)])

        print(tweet_tokens_tokenized)
        print(token_span_info_processed)
        print(offsets, medication_text)
        print()
    print(i)
    print(missing)

    return tweet_io_inputs


def process_tweets_ner(tweets):
    tweets_processed_nomedication = {}
    tweets_processed_medication = {}

    for tweet in tweets:
        if tweet[6] != "-":
            tweets_processed_medication = add_tweet_info(
                tweets_processed_medication, tweet)
        else:
            tweets_processed_nomedication = add_tweet_info(
                tweets_processed_nomedication, tweet)

    ##### Some tweets may contain more than one medication #####
    tweets_processed_medication.update(tweets_processed_nomedication)
    ner_input = io_tagging_process(tweets_processed_medication)

    return ner_input


def input_process():
    ####### 88,988 tweets (only 218 tweets mentioning at least one drug)
    input_1 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.0.tsv")
    input_2 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.1.tsv")
    train = input_1[1:] + input_2[1:]
    train_processed = process_tweets_ner(train)
    print(len(train), len(train_processed))
    read.save_in_tsv("data/bert/ner/smm4h20/train.tsv", train_processed)

    ###### 38,137 tweets (only 93 tweets mentioning at least one drug)
    dev = read.read_from_tsv("data/raw/BioCreative_ValTask3.tsv")
    dev_processed = process_tweets_ner(dev[1:])
    print(len(dev), len(dev_processed))
    read.save_in_tsv("data/bert/ner/smm4h20/dev.tsv", dev_processed)


input_process()
