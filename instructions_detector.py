import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import PROMPT_GUARD_MODEL, DEVICE

class Detector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PROMPT_GUARD_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_MODEL)
        self.model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE
        print(f"Модель детектора загружена и готова к работе.")

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
        return result["label"] != "SAFE" and result["confidence"] >= 0.9
