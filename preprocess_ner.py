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


