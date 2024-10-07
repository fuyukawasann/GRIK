from transformers import AutoMdoelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

model_id = "sshleifer/distilbart-cnn-12-6"
model = AutoMdoelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_onnx = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
