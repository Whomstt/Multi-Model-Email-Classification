import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
import re

# Classifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)

# 1. Load dataset
df = pd.read_csv("AppGallery.csv")

# Convert the dtype object to unicode string
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

# Optional: Rename variable names for easier reference
df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]
df["x"] = df['Interaction content']

df["y"] = df["y2"]

# Remove empty y
df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
print(df.shape)

# 2. Data Grouping
temp = df.copy()
y = temp.y.to_numpy()

# 3. Translation
def trans_to_en(texts):
    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid", download_method=DownloadMethod.REUSE_RESOURCES)

    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l.append(text)
            continue

        doc = nlp_stanza(text)
        lang = doc.lang if doc.lang != "fro" else "fr"  # Handle specific language codes
        if lang != "en":
            tokenizer.src_lang = lang
            encoded = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en"))
            text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        else:
            text_en = text

        text_en_l.append(text_en)
    return text_en_l

# Translate ticket summary (instead of the whole interaction content)
temp["ts_en"] = trans_to_en(temp["Ticket Summary"].tolist())

# 4. Noise Removal
temp["ts"] = temp["Ticket Summary"].str.lower().replace(r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)", " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()

# More noise removal on interaction content
noise_patterns = [
    r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
    r"(january|february|march|april|may|june|july|august|september|october|november|december)",
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"\d{2}(:|.)\d{2}",
    r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
    r"\d+", r"[^0-9a-zA-Z]+", r"(\s|^).(\s|$)"
]

for noise in noise_patterns:
    temp["ic"] = temp["Interaction content"].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()

# Remove less frequent labels
good_y1 = temp.y1.value_counts()[temp.y1.value_counts() > 10].index
temp = temp.loc[temp.y1.isin(good_y1)]
print(temp.shape)

# 5. Textual Data Representation
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
x1 = tfidfconverter.fit_transform(temp["Interaction content"]).toarray()
x2 = tfidfconverter.fit_transform(temp["ts_en"]).toarray()
X = np.concatenate((x1, x2), axis=1)

# 6. Dealing with Data Imbalance
y_series = pd.Series(y)
good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
y = temp["y"]  # Ensure y is defined
y_good = y[y_series.isin(good_y_value)]
X_good = X[y_series.isin(good_y_value)]
y_bad = y[y_series.isin(good_y_value) == False]
X_bad = X[y_series.isin(good_y_value) == False]

test_size = X.shape[0] * 0.2 / X_good.shape[0]
print(f"new_test_size: {test_size}")

X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=test_size, random_state=0)
X_train = np.concatenate((X_train, X_bad), axis=0)
y_train = np.concatenate((y_train, y_bad), axis=0)


