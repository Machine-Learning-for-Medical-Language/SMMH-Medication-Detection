import re

from nltk.tokenize import TweetTokenizer


def add_tweet_info(tweet_dict, info):
    tweet_id, user_id, created_at, text, start, end, span, drug = info[:8]
    if start != "-":
        start = int(start)
        end = int(end)
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


def collect_tweets(tweets):
    tweets_processed_nomedication = {}
    tweets_processed_medication = {}

    for tweet in tweets:
        if tweet[6] != "-" and len(tweet[6]) > 0:
            tweets_processed_medication = add_tweet_info(
                tweets_processed_medication, tweet)
        else:
            tweets_processed_nomedication = add_tweet_info(
                tweets_processed_nomedication, tweet)
    # for key, value in tweets_processed_medication.items():
    #     print(value["text"], value["offsets"], value["span"])
    #     print()
    tweets_processed_medication.update(tweets_processed_nomedication)

    return tweets_processed_medication


def collect_tweets_2018(tweets):
    tweets_processed_nomedication = {}
    tweets_processed_medication = {}

    for tweet in tweets:
        if len(tweet[6]) > 0:
            tweets_processed_medication = add_tweet_info(
                tweets_processed_medication, tweet)
        else:
            tweets_processed_nomedication = add_tweet_info(
                tweets_processed_nomedication, tweet)
    # for key, value in tweets_processed_medication.items():
    #     print(value["text"], value["offsets"], value["span"])
    #     print()
    tweets_processed_medication.update(tweets_processed_nomedication)

    return tweets_processed_medication


def get_spans(tokens, txt):
    span_info = []
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        span_info.append((token, offset, offset + len(token)))
        offset += len(token)
    return span_info


def preprocess(tweets_from_tsv, ner=False):

    tweet_dict = collect_tweets(tweets_from_tsv)

    bert_inputs = []
    i = 0
    missing = []

    tweet_tokenizer = TweetTokenizer()
    tokenizer = re.compile(r"#+|\w+|\S")

    for tweet_id, tweet_info in tweet_dict.items():
        tweet_text = tweet_info["text"]
        offsets = tweet_info["offsets"]
        medication_text = tweet_info["span"]

        # offset_index = []
        # if "-" not in offsets[0][0]:
        #     for item in offsets:
        #         offset_index += [
        #             i for i in range(int(item[0]),
        #                              int(item[1]) + 1)
        #         ]
        tweet_tokens_tokenized = tweet_tokenizer.tokenize(tweet_text)
        tweet_tokens_tokenized_1 = []
        # if tweet_id == "403677548447404032":
        #     print(tweet_id)
        for token in tweet_tokens_tokenized:
            if "#" == token[0]:
                token = [m.group() for m in tokenizer.finditer(token)]
                tweet_tokens_tokenized_1 += token
            elif " " in token:
                tweet_tokens_tokenized_1 += re.split(' +', token)
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

        tweet_label = ["O"] * len(tweet_input)
        if offsets[0][0] != "-":
            for (start_offset, end_offset) in offsets:
                # if start_offset == 129 and end_offset == 135:
                #     print(1)
                for token_idx, (_, token_start_offset, token_end_offset
                                ) in enumerate(token_span_info_processed):
                    if token_start_offset <= start_offset and start_offset < token_end_offset:
                        tweet_label[token_idx] = "B-Medication"
                    elif token_start_offset > start_offset and token_start_offset < end_offset:
                        tweet_label[token_idx] = "I-Medication"

        if len(tweet_label) != len(tweet_input):
            raise ValueError("issues with tokenizations")

        if "B-Medication" in tweet_label:
            label = "1"
        else:
            label = "0"

        # if "I-Medication" not in tweet_label:
        #     # raise Error("issues with tokenizations")
        #     i += 1
        #     missing.append([medication_text, tweet_text])
        if ner is False:
            bert_inputs.append([label, " ".join(tweet_input)])
        else:
            bert_inputs.append([" ".join(tweet_label), " ".join(tweet_input)])

    #     print(tweet_tokens_tokenized)
    #     print(token_span_info_processed)
        for medication_text_single in medication_text:
            print("Keyword: " + medication_text_single)
            print("Tweet: " + " ".join(tweet_input))
            print()

    #     print()
    # print(i)
    # print(missing)

    return bert_inputs
