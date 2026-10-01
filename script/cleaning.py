import pandas as pd
import numpy as np
df = pd.read_csv(r'https://raw.githubusercontent.com/nguyenMinhThanh-dev/netflix_data_cleaning/refs/heads/main/dataset/netflix_titles.csv')
print(df.head())
print(df.info())

#Check duplicate for show_id
print('Total duplicated: ',df['show_id'].duplicated().sum())

#Check null value
print('Null value check: ', df.isna().sum())

#Handle colunms with missing values
#Handle director column

# --- STEP 1: BUILD MAPPING TABLE (ACTOR -> FREQUENTLY COLLABORATED DIRECTORS) ---
# Select only the rows containing complete information about both the cast and the director to train the model
known_df = df[df['director'].notnull() & df['cast'].notnull()].copy()

# Split the list of actors (comma-separated) into separate lines for each actor
actors_df = (
    known_df[['cast', 'director']]
    .assign(actor = df['cast'].str.split(', '))
    .explode('actor')
)

# Count the number of collaborations between each actor and director
pair_count = (
    actors_df.groupby(['actor', 'director'])
    .size()
    .reset_index(name = 'work_count')
)

# Fillter the pairs that have worked together at least twice
frequent_pairs = pair_count[pair_count['work_count'] >= 2]

#Create a dictionary map: {actor : director}
actor_to_director = (
    frequent_pairs.sort_values(by = 'work_count', ascending = False)
    .drop_duplicates(subset = ['actor'])
    .set_index('actor')['director']
    .to_dict()
)

# --- STEP 2: AUTOMATICALLY IMPUTE MISSING DATA ---
def infer_director(row):
    if pd.notnull(row['director']):
        return row['director']
    if pd.isnull(row['cast']):
        return 'Unknown'
    if pd.notnull(row['cast']):
        for actor in row['cast'].split(', '):
            if actor in actor_to_director:
                return actor_to_director[actor]
    return 'Unknown'
df['director'] = df.apply(infer_director, axis = 1)

#Handle country column
#The steps are the same 
known_df = df[df['country'].notnull()].copy()
directors_df = (
    known_df[['country', 'director']]
    .assign(person = df['director'].str.split(', '))
    .explode('person')
)
pair_count = (
    directors_df.groupby(['person', 'country'])
    .size()
    .reset_index(name = 'count')
)
director_dic = (
    pair_count.sort_values(by = 'count', ascending = False)
    .drop_duplicates(subset = 'person')
    .set_index('person')['country']
    .to_dict()
)

def infer_country(row):
    if pd.notnull(row['country']):
        return row['country']
    if row['director'] != 'Unknown':
        for person in row['director'].split(', '):
            if person in director_dic:
                return director_dic[person]
    return 'Unknown'
df['country'] = df.apply(infer_country, axis = 1)

#Handle cast column
df['cast'] = df['cast'].fillna('Unknown')

#Handle the rest missing column
df = df.dropna(subset=['date_added', 'rating', 'duration'])

#Drop descripstion column cause it have no value for analyst
df = df.drop('description', axis = 1)

#Also I only need 1 country per 1 row for my visualazation
df['country'] = df['country'].str.split(',').str[0].str.strip()

#Format the date_added column
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format = '%B %d, %Y', errors = 'coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

#Duel with string column
str_col = df.select_dtypes(include = ['object', 'string']).columns
for c in str_col:
    df[c] = df[c].str.replace(r'\s+',' ', regex = True).str.strip()

#Last cleaning
df['title'] = df['title'].str.strip()

#Export the file after cleanup.
df.to_csv(r'dataset/netflix_cleaned.csv', index = False, encoding = 'utf-8-sig')
