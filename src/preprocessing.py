import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import stanza


# Updated Function for Data Preprocessing
def preprocess_data(file_name):
    # 1. Load dataset
    df = pd.read_csv(file_name)

    # Drop unused or mostly empty columns
    df = df.drop(columns=["Unnamed: 11"], errors="ignore")

    # Convert dtype object to unicode string for compatibility
    if "Interaction content" in df.columns:
        df["Interaction content"] = df["Interaction content"].fillna("").astype(str)
    if "Ticket Summary" in df.columns:
        df["Ticket Summary"] = df["Ticket Summary"].fillna("").astype(str)

    # Optional: Rename variables for easier reference
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]
    df["x"] = df["Interaction content"]
    df["y"] = df["y2"]

    # Remove rows with missing or empty labels
    df = df.loc[(df["y"].notna()) & (df["y"] != "")]

    # 2. Data Grouping
    temp = df.copy()
    y = temp["y"].to_numpy()

    # 3. Translation
    temp["ts_en"] = trans_to_en(temp["Ticket Summary"].tolist())

    # 4. Noise Removal
    temp["ts"] = (
        temp["Ticket Summary"]
        .str.lower()
        .replace(
            r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)",
            " ",
            regex=True,
        )
        .replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # More noise removal on interaction content
    noise_patterns = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        r"\d+",
        r"[^0-9a-zA-Z]+",
        r"(\s|^).(\s|$)",
    ]

    for noise in noise_patterns:
        temp["ic"] = (
            temp["Interaction content"]
            .str.lower()
            .replace(noise, " ", regex=True)
            .replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Remove less frequent labels
    good_y1 = temp["y1"].value_counts()[temp["y1"].value_counts() > 10].index
    temp = temp.loc[temp["y1"].isin(good_y1)]

    # 5. Textual Data Representation
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidfconverter.fit_transform(temp["Interaction content"]).toarray()
    x2 = tfidfconverter.fit_transform(temp["ts_en"]).toarray()
    X = np.concatenate((x1, x2), axis=1)

    # 6. Dealing with Data Imbalance
    y_series = pd.Series(y)
    good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
    y = temp["y"]
    y_good = y[y_series.isin(good_y_value)]
    X_good = X[y_series.isin(good_y_value)]
    y_bad = y[y_series.isin(good_y_value) == False]
    X_bad = X[y_series.isin(good_y_value) == False]

    test_size = X.shape[0] * 0.2 / X_good.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_good, y_good, test_size=test_size, random_state=0
    )
    X_train = np.concatenate((X_train, X_bad), axis=0)
    y_train = np.concatenate((y_train, y_bad), axis=0)

    # Saving preprocessed data to CSV
    preprocessed_df = pd.DataFrame(X_train)
    preprocessed_df["y"] = y_train
    output_file = file_name.replace(".csv", "_preprocessed.csv")
    preprocessed_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")


# Translation Function
def trans_to_en(texts):
    t2t_m = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)

    # Initialize Stanza pipeline for language identification
    stanza.download("multilingual")  # Ensure multilingual resources are downloaded
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid")

    # Define a language fallback map for unsupported codes
    lang_fallback = {
        "nn": "no",  # Map Norwegian Nynorsk to Norwegian Bokmål
        "fro": "fr",  # Old French to modern French
        # Add more mappings if needed
    }

    text_en_l = []
    for text in texts:
        if text == "":  # Skip empty texts
            text_en_l.append(text)
            continue

        try:
            # Detect language
            doc = nlp_stanza(text)
            lang = doc.lang
            lang = lang_fallback.get(lang, lang)

            # Translate if not English
            if lang != "en":
                tokenizer.src_lang = lang
                encoded = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded, forced_bos_token_id=tokenizer.get_lang_id("en")
                )
                text_en = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
            else:
                text_en = text
        except Exception as e:
            print(f"Error processing text: {text}, {str(e)}")
            text_en = text  # Return the original text if translation fails

        text_en_l.append(text_en)
    return text_en_l


# Process both files
preprocess_data("data\\AppGallery.csv")
preprocess_data("data\\Purchasing.csv")
