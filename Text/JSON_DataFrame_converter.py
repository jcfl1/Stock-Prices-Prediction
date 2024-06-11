import json
import os
import pandas as pd



def extract_df_from_json_tweets_data(path_tweets,
                                    relevant_fields = ['author','text','createdAt'],
                                    relevant_author_fields = ['userName','description']):
    dict_list = []

    # For each JSON
    for filename in os.listdir(path_tweets):
        if filename[-4:] != 'json':
            continue

        with open(os.path.join(path_tweets, filename), 'rt', encoding= 'utf8') as f:
            curr_json_list = json.load(f)

        # For each single tweet in a JSON
        for curr_json in curr_json_list:
            # Extract only relevant fields from tweet
            relevant_json = {k:v for k,v in curr_json.items() if k in relevant_fields}
            
            relevant_json_author = {f'author_{k}':v for k,v in relevant_json['author'].items() if k in relevant_author_fields}

            # Delete semi-structured author field in `relevant_json`
            del relevant_json['author']

            # Merging the two dataframes and specifying original file
            new_dict = {**relevant_json, **relevant_json_author}
            new_dict['src_file'] = filename
            dict_list.append(new_dict)

    df = pd.DataFrame(dict_list)
    return df

if __name__ == '__main__':
    df = extract_df_from_json_tweets_data(os.path.join("Scrapping Tweets","PETR4"))
    print(df.info())
    print(df.columns)
    print(df.head(4))