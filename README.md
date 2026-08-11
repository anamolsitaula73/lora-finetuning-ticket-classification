# Fine-Tuning Mini-Project: Support Ticket Classification
### LoRA fine-tuning of Llama 3.1 8B with Unsloth — Phase 8 documentation

---

## 1. Problem Statement 

**Task:** Convert an unstructured customer support message into a structured JSON ticket with three fields: `category`, `urgency`, and `summary`.

**Why fine-tuning (vs RAG or prompting):** This is a *behavior/format* problem, not a knowledge problem — the model doesn't need new facts, it needs to consistently follow a fixed output schema and internalize a consistent labeling scheme across six categories. Fine-tuning is well-suited to this; RAG would add no value here since there's no external knowledge to retrieve.

**Categories used:** `connectivity`, `billing`, `technical`, `security`, `account`, `feature_question`
**Urgency levels used:** `low`, `medium`, `high`

---

## 2. Data

- **60 total hand-reviewed examples**, split into:
  - `train.json` — 51 examples
  - `val.json` — 9 examples (held out, never seen during training)
- Format: Alpaca-style `{instruction, input, output}`, one JSON object per line.
- Examples were written to cover a mix of categories and urgency levels, with deliberately varied phrasing (short/long messages, casual/formal tone, explicit urgency cues vs implicit ones).
- **Limitation:** ~8-9 examples per category is a small sample for a 6-way classification task — this directly shows up in the evaluation results below.

---

## 3. Method

| Setting | Value |
|---|---|
| Base model | Llama 3.1 8B (Unsloth 4-bit quantized) |
| Fine-tuning method | LoRA (via Unsloth) |
| Trainable parameters | 41,943,040 / 8,072,204,288 (0.52%) |
| LoRA target modules | Attention projection layers (q/k/v/o_proj) |
| Hardware | Google Colab, free-tier Tesla T4 GPU |
| Batch size (effective) | 8 (per-device 2 × grad accumulation 4) |
| Learning rate | 2e-4 |
| Epochs / steps | Two runs compared — see below |

### Two training runs were compared:

**Run 1 — max_steps = 60 (~9 epochs over 51 examples)**
- Final training loss: **0.052**
- Loss dropped from 2.58 → 0.05 over 60 steps — a strong overfitting signal, since near-zero loss on a 51-example set almost always means memorization rather than generalization.

**Run 2 — max_steps = 20 (~3 epochs over 51 examples)**
- Chosen specifically to reduce repetition and overfitting risk after Run 1's result.

---

## 4. Evaluation

Evaluated both runs on the same 3 held-out examples from `val.json` (never seen during training):

| Test complaint | Expected | Run 1 output | Run 2 output |
|---|---|---|---|
| "I'm unable to add new team members, the invite button just doesn't respond." | technical / medium | **account** ❌ / medium ✅ | **technical** ✅ / medium ✅ |
| "I can't reset my password, the reset link in the email isn't working at all." | account / high | **security** ❌ / **medium** ❌ | **security** ❌ / **medium** ❌ |
| "Do you offer a student discount? Just curious before I renew." | billing / low | billing ✅ / low ✅ | billing ✅ / low ✅ |

**Accuracy summary:**
| Metric | Run 1 (60 steps) | Run 2 (20 steps) |
|---|---|---|
| Category accuracy | 1/3 | 2/3 |
| Urgency accuracy | 2/3 | 3/3 |
| Summary quality | Paraphrased, not verbatim, in both runs | Paraphrased, not verbatim, in both runs |

### Key findings

1. **Reducing training steps from 60 to 20 improved held-out accuracy** — direct evidence that the first run had overfit despite (and because of) its very low training loss. This is a concrete, measurable example of the "low training loss ≠ better model" lesson.
2. **Summarization generalized well in both runs** — the model consistently produced original phrasing rather than copying training examples verbatim, even in the overfit run. This suggests summarization is an easier sub-skill to pick up than precise category boundaries with this little data.
3. **Persistent failure case: account vs security ambiguity.** Both runs mislabeled the password-reset complaint the same way, suggesting this isn't a training-duration issue but a data coverage issue — the dataset likely needs more examples that draw a clearer line between "account access" issues and "security/compromise" issues.

---

## 5. Limitations

- 51 training examples is a proof-of-concept scale, not production scale. Real production use of this task would need hundreds to low-thousands of labeled examples per category to get reliable classification accuracy.
- Evaluation used only 3 manually-reviewed examples for the head-to-head comparison; a production evaluation would score the full 9-example (or larger) validation set programmatically (e.g. exact-match on category/urgency, JSON-schema validity rate) rather than manual inspection.
- No comparison against prompting the base model with few-shot examples was performed — a fair production decision would benchmark fine-tuning against that cheaper alternative before committing to a fine-tuned deployment.

---

## 6. What I'd do differently at scale

- Increase examples per category to at least 30-50 each, focusing extra examples on the account/security boundary that caused errors.
- Run a proper automatic evaluation over the full validation set (not just 3 examples) with exact-match scoring.
- Try a small sweep of `max_steps` (e.g. 10, 20, 30, 40) with loss + validation accuracy logged at each, rather than comparing just two arbitrary points.
- Compare against a pure prompting baseline (same base model, few-shot examples in the prompt, no fine-tuning) to check whether fine-tuning was actually necessary for this task.

---

## 7. How to Run

The trained LoRA adapter is hosted on Hugging Face Hub: **[anamolsitaula/llama3-ticket-classifier-lora](https://huggingface.co/anamolsitaula/llama3-ticket-classifier-lora)**

The base model (`unsloth/Meta-Llama-3.1-8B-bnb-4bit`) and the adapter above are both downloaded automatically from Hugging Face — no local model files needed.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "anamolsitaula/llama3-ticket-classifier-lora",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
    [alpaca_prompt.format(
        "Convert this customer message into a structured support ticket with category, urgency, and summary.",
        "Your customer complaint here",
        "",
    )], return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.batch_decode(outputs))
```

**To retrain from scratch instead of just running inference:** open the Unsloth Colab notebook, upload `train.json` from this repo, and follow the same steps documented above (attach LoRA adapters, format with the Alpaca prompt template, train with `SFTTrainer`, `max_steps = 20`).

## 8. Cost / Infra Notes

- Training ran entirely on Google Colab's free-tier T4 GPU — $0 cost.
- Each training run took under 5 minutes given the small dataset size.
- LoRA adapters (not a merged model) were saved locally — deployable by loading the base model + adapter at inference time, without duplicating the full 8B-parameter model.
