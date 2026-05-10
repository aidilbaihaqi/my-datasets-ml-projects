import pandas as pd
import time
from google_play_scraper import Sort, reviews

app_id = "com.shopee.id"

all_reviews = []
token = None

TARGET = 15000

while len(all_reviews) < TARGET:
    batch, token = reviews(
        app_id,
        lang="id",
        country="id",
        sort=Sort.NEWEST,
        count=200,
        continuation_token=token
    )
    
    all_reviews.extend(batch)
    print("Total:", len(all_reviews))

    if token is None:
        break
    
    time.sleep(1)

df_raw = pd.DataFrame(all_reviews)[["content"]]
df_raw = df_raw.dropna().drop_duplicates().head(TARGET)

df_raw.to_csv("raw_data.csv", index=False)
print("Final data:", len(df_raw))