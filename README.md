##  Mistral-7B-OIG-Unslooth Model Inference
Here’s a clean and simplified version of your **`README.md`**:

---

## 🚀 Mistral-7B-OIG-Unslooth Model Inference

This project demonstrates how to load and perform inference using the `Praveen9121/mistral-7b-oig-unsloth-merged` model with Hugging Face's `transformers` library.

---

## 📦 **Required Packages**
To run the inference, make sure you have the following packages installed:
```bash
pip install torch transformers
```

---

## 🧠 **Model Details**
- **Model Name:** `Praveen9121/mistral-7b-oig-unsloth-merged`
- **Base Model:** Mistral-7B
- **Fine-Tuned Dataset:** OIG + Unslooth dataset

---

## 📝 **Inference Script**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Praveen9121/mistral-7b-oig-unsloth-merged")
model = AutoModelForCausalLM.from_pretrained("Praveen9121/mistral-7b-oig-unsloth-merged")

# Create text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input text for generation
input_text = "Explain the importance of AI in education."

# Generate text
output = pipe(input_text, 
              max_length=200,
              num_return_sequences=1,
              temperature=0.7,
              top_p=0.9,
              do_sample=True
             )

# Print the generated text
print("Generated Text:\n", output[0]['generated_text'])
```

---

## 🎯 **Usage Instructions**
1. Save the above code in a file named `infer.py`.
2. Run the script:
```bash
python infer.py
```

---



### Issue: CUDA not detected
- Check if PyTorch is installed with GPU support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---
