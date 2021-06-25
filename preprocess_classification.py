import read_files as read
from tokenization import collect_tweets, preprocess

###### field = [tweet_id	user_id	created_at	text	start	end	span	drug]
######   empty value "-"

# mapping_rules = []

##### Some tweets may contain more than one medication #####


def input_process():
    ####### 88,988 tweets (only 218 tweets mentioning at least one drug)
    input_1 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.0.tsv")
    input_2 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.1.tsv")
    train = input_1[1:] + input_2[1:]

    train_processed = preprocess(train, ner=False)

    print(len(train_processed))

    read.save_in_tsv(
        "data/bert/classifier/smm4h20+_nertoclassifer_smallset/train.tsv",
        train_processed[:1000])

    read.save_in_tsv("data/bert/classifier/smm4h20+_nertoclassifer/train.tsv",
                     train_processed)

    ###### 38,137 tweets (only 93 tweets mentioning at least one drug)
    dev = read.read_from_tsv("data/raw/BioCreative_ValTask3.tsv")
    dev_processed = preprocess(dev[1:], ner=False)
    print(len(dev_processed))

    read.save_in_tsv(
        "data/bert/classifier/smm4h20+_nertoclassifer_smallset/dev.tsv",
        dev_processed[:1000])

    read.save_in_tsv("data/bert/classifier/smm4h20+_nertoclassifer/dev.tsv",
                     dev_processed)


input_process()
