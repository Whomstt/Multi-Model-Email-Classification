import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
import stanza
from stanza.pipeline.core import DownloadMethod
import re
import os


# Preprocessing function
def preprocess_dataset(df, content_col, summary_col):
    # Ensure columns are treated as strings
    df[content_col] = df[content_col].values.astype("U")
    df[summary_col] = df[summary_col].values.astype("U")

    # Translation
    def trans_to_en(texts):
        t2t_m = "facebook/m2m100_418M"
        model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
        tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
        nlp_stanza = stanza.Pipeline(
            lang="multilingual",
            processors="langid",
            download_method=DownloadMethod.REUSE_RESOURCES,
        )

        translated_texts = []
        for text in texts:
            if not text.strip():
                translated_texts.append(text)
                continue

            # Detect language
            doc = nlp_stanza(text)
            lang = doc.lang if doc.lang != "fro" else "fr"
            if lang != "en":
                tokenizer.src_lang = lang
                encoded = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded, forced_bos_token_id=tokenizer.get_lang_id("en")
                )
                translated_text = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
            else:
                translated_text = text

            translated_texts.append(translated_text)
        return translated_texts

    df["ts_en"] = trans_to_en(df[summary_col].tolist())

    # Noise removal
    noise_patterns = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)|(fw(d)?\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"(\s|^).(\s|$)",
    ]
    for noise in noise_patterns:
        df[content_col] = (
            df[content_col]
            .str.lower()
            .replace(noise, " ", regex=True)
            .replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["ts_en"] = (
            df["ts_en"]
            .str.lower()
            .replace(noise, " ", regex=True)
            .replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # TF-IDF Transformation
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x_content = tfidfconverter.fit_transform(df[content_col]).toarray()
    x_summary = tfidfconverter.fit_transform(df["ts_en"]).toarray()
    X = np.concatenate((x_content, x_summary), axis=1)

    return X


# Load datasets
appgallery_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "AppGallery.csv"
)
purchasing_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "Purchasing.csv"
)

df_appgallery = pd.read_csv(appgallery_path)
df_purchasing = pd.read_csv(purchasing_path)

# Preprocess both datasets
X_appgallery = preprocess_dataset(
    df_appgallery, "Interaction content", "Ticket Summary"
)
X_purchasing = preprocess_dataset(
    df_purchasing, "Interaction content", "Ticket Summary"
)

# Combine datasets
X_combined = np.concatenate((X_appgallery, X_purchasing), axis=0)

# Output shape
print("Preprocessed feature matrix shape:", X_combined.shape)
