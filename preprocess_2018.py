from collections import Counter

import read_files as read

###### field = [tweet_id	user_id	created_at	text	start	end	span	drug]
######   empty value "-"


def extract_positive(train_processed):
    medication_list_2021 = read.read_from_json(
        "data/processed/medicaions_2021")
    medication_list_2021 = [item.lower() for item in medication_list_2021]

    medication_list = []
    medication_variant = {}
    negative_tweet = []
    positive_tweet_found = []
    positive_tweet_not_found = []

    positive_tweet_found_subset = []

    tweet_mistakes = 0
    positive_tweets = 0
    for item in train_processed:
        if len(item) == 7 and len(item[5]) > 0:
            [
                tweet_id, user_id, text, created_at, label, medication,
                medication_norm
            ] = item
            medication_list = medication.split("; ")
            medication_norm_list = medication_norm.split("; ")
            for medication_single, medication_single_norm in zip(
                    medication_list, medication_norm_list):
                positive_tweets += 1

                medication_single_list = medication_single.split(" ")

                if len(medication_single_list) == 2:
                    medication_single_list_1lower_2cap = medication_single_list[
                        0].lower(
                        ) + " " + medication_single_list[1].capitalize()
                    medication_single_list_1cap_2lower = medication_single_list[
                        0].capitalize(
                        ) + " " + medication_single_list[1].lower()

                medication_single_list_cap = " ".join(
                    [item.capitalize() for item in medication_single_list])

                medication_single_lower = medication_single.lower()

                medication_single_cap = medication_single.capitalize()
                medication_single_upper = medication_single.upper()

                medication_in_text = ""

                if medication_single in text:
                    start = text.find(medication_single)
                    end = start + len(medication_single)
                    medication_in_text = medication_single
                elif medication_single_list_cap in text:
                    start = text.find(medication_single_list_cap)
                    end = start + len(medication_single_list_cap)
                    medication_in_text = medication_single_list_cap
                elif medication_single_cap in text:
                    start = text.find(medication_single_cap)
                    end = start + len(medication_single_cap)
                    medication_in_text = medication_single_cap
                elif medication_single_lower in text:
                    start = text.find(medication_single_lower)
                    end = start + len(medication_single_lower)
                    medication_in_text = medication_single_lower
                elif medication_single_list_1lower_2cap in text:
                    start = text.find(medication_single_list_1lower_2cap)
                    end = start + len(medication_single_list_1lower_2cap)
                    medication_in_text = medication_single_list_1lower_2cap
                elif medication_single_list_1cap_2lower in text:
                    start = text.find(medication_single_list_1cap_2lower)
                    end = start + len(medication_single_list_1cap_2lower)
                    medication_in_text = medication_single_list_1cap_2lower
                elif medication_single_upper in text:
                    start = text.find(medication_single_upper)
                    end = start + len(medication_single_upper)
                    medication_in_text = medication_single_upper
                else:
                    start, end = None, None
                    medication_in_text = medication_single
                    # print(text, "_____", medication_single, "_____",
                    #       medication)
                    # print()
                    tweet_mistakes += 1

                if start is not None and end is not None:
                    positive_tweet_found.append([
                        tweet_id, user_id, created_at, text, start, end,
                        medication_in_text,
                        medication_single_norm.lower()
                    ])
                    if medication_single_norm.lower(
                    ) in medication_list_2021 or medication_in_text.lower(
                    ) in medication_list_2021:
                        positive_tweet_found_subset.append([
                            tweet_id, user_id, created_at, text, start, end,
                            medication_in_text,
                            medication_single_norm.lower()
                        ])

                else:
                    positive_tweet_not_found.append([
                        tweet_id, user_id, created_at, text, "_", "_",
                        medication_in_text,
                        medication_single_norm.lower()
                    ])

            # for medication, medication_norm in zip()
        else:
            [
                tweet_id, user_id, text, created_at, label, medication,
                medication_norm
            ] = item
            if len(medication) > 0:
                print(medication, medication_norm)
            negative_tweet.append(
                [tweet_id, user_id, created_at, text, "_", "_", "_", "_"])

    print(positive_tweets, tweet_mistakes)

    read.save_in_tsv("data/processed/negative_2018_tweet.tsv", negative_tweet)
    read.save_in_tsv("data/processed/positive_2018_tweet.tsv",
                     positive_tweet_found)
    read.save_in_tsv("data/processed/positive_not_found_2018_tweet.tsv",
                     positive_tweet_not_found)
    read.save_in_tsv("data/processed/positive_found_overlap_2018_tweet.tsv",
                     positive_tweet_found_subset)


def get_medication():
    from tokenization import collect_tweets_2018, preprocess
    input_1 = read.read_from_tsv(
        "data/raw/SMM4H18_Train_revised; add_normalized.csv")

    train = input_1[1:]

    # print(train)

    extract_positive(train)


get_medication()


def upsampling():
    train_input = read.read_from_tsv(
        "data/bert/classifier/smm4h20+_nertoclassifer/train.tsv")
    print(len(train_input))  ###88983

    positive = [item for item in train_input if item[0] == "1"]
    print(len(positive))  ###218
    train_upsampling = positive * 99 + train_input

    # read.save_in_tsv(
    #     "data/bert/classifier/smm4h20+_nertoclassifer_upsampling/train.tsv",
    #     train_upsampling)


# upsampling()
