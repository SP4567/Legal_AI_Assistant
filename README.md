# Legal_AI_Assistant

### 📄 `README.md` — AI Lawyer: Verdict Generator Using Gemini LLM

```markdown
# ⚖️ AI Lawyer: Smart Verdict Generator

AI Lawyer is a cutting-edge legal assistant that analyzes case titles and descriptions to generate detailed legal verdicts using Google’s Gemini LLM. It mimics real-world legal reasoning, citing similar case precedents, and outputs a comprehensive judgment including punishment or relief suggestions.

---

## 🚀 Features

- 🧠 Trained on your custom legal dataset
- 🔍 Searches for similar past cases using semantic search (via embeddings)
- 📜 Generates full verdicts with:
  - Legal reasoning
  - Guilty/Not Guilty/Settled outcomes
  - Suggested punishments and reliefs
- 🤖 Uses Google's Gemini 2.0 API
- 🌐 Interactive UI built using **Streamlit**

---

## 📂 Dataset

- Dataset includes:
  - `case_title`
  - `case_text`
  - LLM-generated `verdict` (during inference)
- Preprocessing includes:
  - Prompt construction
  - Text cleaning
  - Embedding vectorization (for similarity search)

---

## 🛠️ Installation & Setup

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/ai-lawyer.git
cd ai-lawyer
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Set up Google Gemini API**
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY_HERE")
```

4. **Run the app**
```bash
streamlit run app.py
```

---

## 🖥️ UI Preview

![screenshot](preview.png)

---

## 🧠 How It Works

1. Takes a **new legal case input** (title + description)
2. Searches the dataset for **semantically similar cases**
3. Constructs a **contextual prompt** using past cases
4. Sends prompt to **Gemini LLM**
5. Displays **detailed legal verdict**

---

## 📦 File Structure

```
├── app.py                      # Streamlit UI
├── preprocess.py              # Dataset preprocessing & prompt building
├── model_utils.py             # Embedding, vector search, Gemini API logic
├── preprocessed_dataset.csv   # Your clean dataset
├── requirements.txt
└── README.md
```

---

## 🔒 Disclaimer

> This project is a prototype for educational and experimental purposes. It is **not a substitute for licensed legal advice**. The generated verdicts are based on language model outputs, not actual jurisprudence.

---

## 🙌 Credits

Developed with ❤️ by [Suyash Pandey](https://github.com/SP4567)

- Inspired by legal AI systems, fine-tuned for the modern era.
- Backed by the brilliance of Google Gemini + NLP smarts.

---

## 📜 License

MIT License — feel free to fork, remix, and make law less scary.
```

---

If you want me to generate the actual `requirements.txt` or plug in screenshots/diagrams or even a custom badge or logo — I can whip those up too. Let’s make your repo *trial-ready*.
