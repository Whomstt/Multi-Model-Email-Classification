import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from patterns.singleton.TranslationManager import TranslationManager

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

    # 3. Translation (Using TranslationManager)
    translator = TranslationManager()  # Access the singleton
    temp["ts_en"] = translator.translate_to_en(temp["Ticket Summary"].tolist())

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
