import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import PROMPT_GUARD_MODEL_PATH, DEVICE
import re

class Detector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PROMPT_GUARD_MODEL_PATH)
        #self.model = AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_MODEL_PATH)
        self.model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE


    def check(self, text: str) -> dict[str, any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        label = self.model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1).max().item()

        return {"label": label, "confidence": confidence}

    def is_malicious(self, text: str) -> bool:
        result = self.check(text)
        #print(f"{text} \n {result["confidence"]}\t {result['label']}")
        return result["label"] != "SAFE" and result["confidence"] >= 0.996


    def has_malicious_fragment(self, text: str) -> bool:
        segments = re.split(r'[.!?\[\]()]+', text)
        segments = [s.strip() for s in segments if s.strip()]

        for segment in segments:
            if self.is_malicious(segment):
                return True
            token_count = len(self.tokenizer.encode(segment, add_special_tokens=False))
            if token_count > 40 and self.sliding_window_check(segment):
                return True

        return False

    def sliding_window_check(self, text: str) -> bool:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        window_size = 40
        window_stride = 20

        if len(token_ids) <= window_size:
            return False

        for start in range(0, len(token_ids), window_stride):
            window_ids = token_ids[start:start + window_size]
            if not window_ids:
                break

            window_text = self.tokenizer.decode(window_ids)
            if self.is_malicious(window_text):
                return True

            if start + window_size >= len(token_ids):
                break

        return False
