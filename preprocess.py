import read_files as read

###### field = [tweet_id	user_id	created_at	text	start	end	span	drug]
######   empty value "-"


def process_tweets(tweets):
    tweets_processed_nomedication = []
    tweets_processed_medication = []

    for tweet in tweets:
        if tweet[6] != "-":
            tweets_processed_medication.append(tweet[3].rstrip())
        else:
            tweets_processed_nomedication.append(tweet[3].rstrip())

    ##### Some tweets may contain more than one medication #####
    tweets_processed_medication = list(set(tweets_processed_medication))
    print(len(tweets_processed_medication))

    tweets_processed = [["1", item] for item in tweets_processed_medication
                        ] + [["0", item]
                             for item in tweets_processed_nomedication]

    return tweets_processed


def input_process():
    ####### 88,988 tweets (only 218 tweets mentioning at least one drug)
    input_1 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.0.tsv")
    input_2 = read.read_from_tsv("data/raw/BioCreative_TrainTask3.1.tsv")
    train = input_1[1:] + input_2[1:]
    train_processed = process_tweets(train)
    print(len(train), len(train_processed))
    read.save_in_tsv("data/bert/classifier/smm4h20+1/train.tsv",
                     train_processed[:500])

    ####### 38,137 tweets (only 93 tweets mentioning at least one drug)
    dev = read.read_from_tsv("data/raw/BioCreative_ValTask3.tsv")
    dev_processed = process_tweets(dev[1:])
    print(len(dev), len(dev_processed))
    read.save_in_tsv("data/bert/classifier/smm4h20+1/dev.tsv",
                     dev_processed[:500])


input_process()
