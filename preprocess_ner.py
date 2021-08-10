import read_files as read
from tokenization import preprocess

###### field = [tweet_id	user_id	created_at	text	start	end	span	drug]
######   empty value "-"

# mapping_rules = []


def input_process():
    ####### 88,988 tweets (only 218 tweets mentioning at least one drug)
    input_1 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.0.tsv")
    input_2 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.1.tsv")
    train = input_1[1:] + input_2[1:]

    train_processed = preprocess(train, ner=True)

    print(len(train_processed))
    read.save_in_tsv("data/bert/ner/smm4h20+_smallset/train.tsv",
                     train_processed[:1000])

    read.save_in_tsv("data/bert/ner/smm4h20+/train.tsv", train_processed)

    ###### 38,137 tweets (only 93 tweets mentioning at least one drug)
    dev = read.read_from_tsv("data/raw/BioCreative_ValTask3.tsv")
    dev_processed = preprocess(dev[1:], ner=True)

    print(len(dev_processed))
    read.save_in_tsv("data/bert/ner/smm4h20+_smallset/dev.tsv",
                     dev_processed[:1000])

    read.save_in_tsv("data/bert/ner/smm4h20+/dev.tsv", dev_processed)


# input_process()


def input_process_2018():
    ####### 88,988 tweets (only 218 tweets mentioning at least one drug)
    input_1 = read.read_from_tsv("data/processed/positive_2018_tweet.tsv")

    positive_2018_tweet = preprocess(input_1, ner=True)

    print(len(positive_2018_tweet))

    read.save_in_tsv("data/bert/ner/smm4h18_positive_ner/train.tsv",
                     positive_2018_tweet)

    input_2 = read.read_from_tsv(
        "data/processed/positive_found_overlap_2018_tweet.tsv")

    positive_2018_tweet = preprocess(input_2, ner=True)

    print(len(positive_2018_tweet))

    read.save_in_tsv("data/bert/ner/smm4h18_positive_overlap_ner/train.tsv",
                     positive_2018_tweet)


# input_process_2018()


def input_process_2018():
    input_1 = read.read_from_tsv("data/processed/positive_2018_tweet.tsv")
    input_2 = read.read_from_tsv(
        "data/processed/positive_not_found_2018_tweet_annotate_position.tsv")
    input_3 = read.read_from_tsv("data/processed/negative_2018_tweet.tsv")

    positive_2018_tweet = preprocess(input_1 + input_2, ner=True)

    print(len(positive_2018_tweet))
    read.save_in_tsv("data/bert/ner/smm4h18_positive_all_ner/train.tsv",
                     positive_2018_tweet)


# input_process_2018()


def process_text(text):
    text = text.replace("\n", " ")
    return text


def process_top200():
    # import pandas as pd
    # data = pd.read_csv('data/raw/top200_en.csv')

    # def clean(row):
    #     return row['text'].replace("\n", "")

    # data['clean'] = data.apply(clean, axis=1)
    # data.drop(['text'], axis=1)
    # data.rename(columns={'clean': 'text'})
    # data = data[[
    #     'tweet_id', 'user_id', 'created_at', 'text', 'start', 'end', 'span',
    #     'drug'
    # ]]
    # data.to_csv("data/raw/top200_en.tsv", sep='\t', encoding='utf-8')

    data = read.read_from_csv("data/raw/top200_en.csv")
    input_new = []
    for item in data[1:]:
        item = item[1:]
        item[3] = process_text(item[3])
        input_new.append(item)

    positive_top_200_tweet = preprocess(input_new, ner=True)
    # print(positive_top_200_tweet[:10])

    read.save_in_tsv("data/bert/ner/top200_positive/train.tsv",
                     positive_top_200_tweet)


process_top200()
