"""
Retail Banking Assistant - standalone Gradio prototype.

Loads the fine-tuned FLAN-T5 model saved by the notebook (the `final_model` folder) and serves a 3-tab app:
  1) Chat   2) Evaluate a response (live ROUGE/BLEU/METEOR)   3) Model report

Run locally / in Colab:   MODEL_DIR=./final_model python app.py
Hugging Face Space:       upload this file + requirements.txt + the final_model folder (or set MODEL_DIR to a Hub repo id).
"""
import os, re, json, time, unicodedata
import torch
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = os.environ.get("MODEL_DIR", "./final_model")     # local folder saved by the notebook OR a Hugging Face repo id
PREFIX = "Answer the retail banking customer query: "
DEFAULT_GEN = dict(num_beams=3, no_repeat_ngram_size=3, repetition_penalty=1.3, do_sample=False)
MAX_SRC, MAX_TGT = 128, 256

# --- read the settings the notebook found best (if run_summary.json is next to the model) -----------------------------
SUMMARY = {}
_summary_path = os.path.join(MODEL_DIR, "run_summary.json")
if os.path.exists(_summary_path):
    SUMMARY = json.load(open(_summary_path))
    DEFAULT_GEN = {**DEFAULT_GEN, **SUMMARY.get("best_decoding_config", {})}
    MAX_SRC = int(SUMMARY.get("max_source_tokens", MAX_SRC)); MAX_TGT = int(SUMMARY.get("max_target_tokens", MAX_TGT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device).eval()

# --- optional metric libraries (the Evaluate tab degrades gracefully if they are missing) -----------------------------
try:
    from rouge_score import rouge_scorer
    import sacrebleu, nltk
    from nltk.translate.meteor_score import meteor_score
    for pkg in ("wordnet", "omw-1.4"):
        try: nltk.download(pkg, quiet=True)
        except Exception: pass
    _rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    METRICS_OK = True
except Exception:
    METRICS_OK = False


def clean_text(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", s).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"([!?.,])\1{2,}", r"\1", s)
    return re.sub(r"\s+([,.!?;:])", r"\1", s)


def trigram_repetition(text):
    w = text.lower().split()
    if len(w) < 4: return 0.0
    tri = list(zip(w, w[1:], w[2:]))
    return 1 - len(set(tri)) / len(tri)


def answer(question, num_beams=None, no_repeat=None, rep_pen=None, temperature=0.0, max_tokens=None):
    kw = dict(DEFAULT_GEN)
    if num_beams is not None: kw["num_beams"] = int(num_beams)
    if no_repeat is not None: kw["no_repeat_ngram_size"] = int(no_repeat)
    if rep_pen is not None:   kw["repetition_penalty"] = float(rep_pen)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=float(temperature), top_p=0.95, num_beams=1)
    else:
        kw.update(do_sample=False); kw.pop("temperature", None); kw.pop("top_p", None)
    enc = tokenizer(PREFIX + clean_text(question), return_tensors="pt", truncation=True, max_length=MAX_SRC).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=int(max_tokens or MAX_TGT), **kw)
    return tokenizer.decode(out[0], skip_special_tokens=True), time.time() - t0


def respond(message, history, nb, nr, rp, temp, mt):
    history = history or []
    if not message or not message.strip():
        return history, "", ""
    ans, dt = answer(message, nb, nr, rp, temp, mt)
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ans}]
    return history, "", f"⏱️ {dt * 1000:.0f} ms"


def evaluate_live(question, reference):
    if not question.strip() or not reference.strip():
        return "Please enter both a question and a reference answer.", {}
    pred, dt = answer(question)
    out = {"Length ratio": round(len(pred.split()) / max(1, len(reference.split())), 3),
           "Repetition rate": round(trigram_repetition(pred), 4), "Latency (ms)": round(dt * 1000)}
    if METRICS_OK:
        sc = _rouge.score(reference, pred)
        try:    met = float(meteor_score([reference.split()], pred.split()))
        except Exception: met = float("nan")
        out = {"ROUGE-1": round(sc["rouge1"].fmeasure, 4), "ROUGE-2": round(sc["rouge2"].fmeasure, 4),
               "ROUGE-L": round(sc["rougeL"].fmeasure, 4),
               "BLEU": round(sacrebleu.sentence_bleu(pred, [reference]).score / 100, 4), "METEOR": round(met, 4), **out}
    else:
        out["note"] = "Install rouge_score, sacrebleu and nltk to see ROUGE/BLEU/METEOR."
    return pred, out


def load_report():
    p = os.path.join(MODEL_DIR, "evaluation_results.csv")
    if os.path.exists(p):
        return pd.read_csv(p).rename(columns={"Unnamed: 0": "model"}).round(4)
    return pd.DataFrame({"info": ["evaluation_results.csv not found next to the model - run the notebook first."]})


def make_chatbot():
    try:    return gr.Chatbot(type="messages", height=430, label="Banking Assistant")
    except TypeError: return gr.Chatbot(height=430, label="Banking Assistant")


EXAMPLES = ["How can I activate my debit card?", "I lost my credit card, what should I do?", "How do I check my account balance?",
            "where is the nearest atm", "I want to apply for a personal loan", "how do i dispute a charge on my statement",
            "What documents do I need to open a savings account?"]


def build_demo():
    with gr.Blocks(title="Retail Banking Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏦 Retail Banking Assistant\nFine-tuned **FLAN-T5** on the Bitext retail-banking dataset. "
                    "Demo only - do not enter real account details; answers are not financial advice.")
        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                chatbot = make_chatbot()
                msg = gr.Textbox(label="Your banking question", placeholder="e.g. How can I activate my debit card?")
                with gr.Row():
                    send = gr.Button("Send", variant="primary"); gr.ClearButton([msg, chatbot], value="Clear")
                latency = gr.Markdown()
                with gr.Accordion("⚙️ Generation settings", open=False):
                    nb = gr.Slider(1, 8, value=DEFAULT_GEN.get("num_beams", 1), step=1, label="Beam search width")
                    nr = gr.Slider(0, 5, value=DEFAULT_GEN.get("no_repeat_ngram_size", 0), step=1, label="No-repeat n-gram size (0 = off)")
                    rp = gr.Slider(1.0, 2.0, value=float(DEFAULT_GEN.get("repetition_penalty", 1.0)), step=0.05, label="Repetition penalty")
                    temp = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature (0 = deterministic)")
                    mt = gr.Slider(32, MAX_TGT, value=min(256, MAX_TGT), step=16, label="Max new tokens")
                gr.Examples(examples=EXAMPLES, inputs=msg, label="Try these")
                ins = [msg, chatbot, nb, nr, rp, temp, mt]
                msg.submit(respond, ins, [chatbot, msg, latency]); send.click(respond, ins, [chatbot, msg, latency])
            with gr.Tab("🧪 Evaluate a response"):
                gr.Markdown("Enter a customer question **and** the ideal answer; the model replies and metrics are computed live.")
                q_in = gr.Textbox(label="Customer question"); ref_in = gr.Textbox(label="Reference (ideal) answer", lines=4)
                run = gr.Button("Generate & evaluate", variant="primary")
                pred_out = gr.Textbox(label="Model answer", lines=6); met_out = gr.JSON(label="Metrics")
                run.click(evaluate_live, [q_in, ref_in], [pred_out, met_out])
            with gr.Tab("📊 Model report"):
                gr.Markdown("Held-out test-set results saved by the notebook (zero-shot vs fine-tuned vs tuned decoding).")
                gr.Dataframe(value=load_report(), interactive=False)
                if SUMMARY:
                    gr.Markdown(f"**Best training hyper-parameters:** `{SUMMARY.get('best_training_hyperparameters')}`  \n"
                                f"**Best decoding config:** `{SUMMARY.get('best_decoding_config')}`")
    return demo


if __name__ == "__main__":
    build_demo().queue().launch(share=os.environ.get("SHARE", "0") == "1")
