from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import stanza

class TranslationManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_resources()
        return cls._instance

    def _initialize_resources(self):
        print("Initializing Translation Model and Language Identification Pipeline...")
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        stanza.download("multilingual", processors="langid")  # Ensure resources are downloaded
        self.nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid")
        self.lang_fallback = {"nn": "no", "fro": "fr"}  # Example: map old languages to modern equivalents

    def translate_to_english(self, texts):
        translated_texts = []
        for text in texts:
            if not text:
                translated_texts.append(text)
                continue

            try:
                # Detect language
                doc = self.nlp_stanza(text)
                lang = self.lang_fallback.get(doc.lang, doc.lang)

                # Translate if not English
                if lang != "en":
                    self.tokenizer.src_lang = lang
                    encoded = self.tokenizer(text, return_tensors="pt")
                    generated_tokens = self.model.generate(
                        **encoded, forced_bos_token_id=self.tokenizer.get_lang_id("en")
                    )
                    text_en = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                else:
                    text_en = text
            except Exception as e:
                print(f"Translation error for text: {text}, {str(e)}")
                text_en = text  # Fallback to original text if translation fails

            translated_texts.append(text_en)

        return translated_texts