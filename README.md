# Sentinel — Local Desktop AI Agent

> *"Jarvis for your Windows PC — completely offline, no API keys required."*

Sentinel is an intelligent desktop operating system assistant you control by typing natural‐language commands.  It understands your intent, builds a multi-step execution plan, carries it out, and explains every action in real time.

---

## What Sentinel can do

| Category | Commands |
|---|---|
| **Apps** | Open Chrome / Rider / VS Code / GitHub Desktop / Outlook / Task Manager / Notepad… |
| **Files** | Search PDFs modified today · Create folders · Move / copy / rename / delete files · Summarise a text file |
| **Memory** | *"Remember that my favourite IDE is Rider"* → *"Open my IDE"* |
| **System** | Volume up/down/mute · Screenshot · System info · Shutdown in N minutes · Sleep / restart |
| **Web** | Search the web (opens browser) |
| **Help** | *"What can you do?"* |

---

## Architecture

```
sentinel/
├── core/
│   ├── llm/
│   │   ├── model.py        ← GPT-2 style transformer (scratch, ~15 M params)
│   │   ├── tokenizer.py    ← BPE tokeniser (scratch, no sentencepiece)
│   │   └── trainer.py      ← AMP training loop, cosine warmup, early stop
│   ├── agent/
│   │   ├── agent.py        ← Central facade  (classify → plan → execute)
│   │   ├── classifier.py   ← Neural + rule-based hybrid classifier
│   │   ├── planner.py      ← Converts intent → ExecutionPlan (steps)
│   │   ├── executor.py     ← Runs steps, fires UI callbacks
│   │   └── memory.py       ← SQLite: preferences, facts, history
│   └── tools/
│       ├── apps.py         ← Launch / close Windows applications
│       ├── files.py        ← File search, CRUD, TF-IDF summarisation
│       └── system.py       ← Volume, screenshot, shutdown, system info
├── data/
│   └── training_data.py    ← 25 intents, 200 + labelled examples
├── ui/
│   └── window.py           ← PyQt5 Jarvis-style interface
├── main.py                 ← Application entry point
├── train.py                ← Model training script
└── requirements.txt
```

Every major component is **swappable**.  You can replace the transformer with a larger model you train yourself; the rest of the system doesn't change.

---

## Requirements

- **Windows 10 / 11** (the tool layer uses Win32 APIs)
- **Python 3.10+**
- **NVIDIA GPU** with CUDA 12.x (RTX 3050 4 GB works perfectly)
- ~2 GB disk space for PyTorch + model checkpoint

---

## Installation

### 1 — Clone / copy the project

```
cd C:\Users\<you>\Projects
# (place the sentinel/ folder here)
cd sentinel
```

### 2 — Create a virtual environment (recommended)

```
py -m venv venv
venv\Scripts\activate
```

### 3 — Install PyTorch with CUDA 12.1

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **CPU-only fallback** (slower, but works):
> ```
> pip install torch torchvision
> ```

### 4 — Install remaining dependencies

```
pip install -r requirements.txt
```

---

## Training the model

Training teaches Sentinel to understand commands using your GPU.  It only needs to be done once (or whenever you add new intents / examples).

```
python train.py
```

**What happens:**
1. Trains a BPE tokeniser on the example corpus
2. Builds a 15 M-parameter transformer
3. Trains for up to 50 epochs with AMP (mixed precision) + cosine warm-up
4. Saves the best checkpoint to `checkpoints/`

Expected time on RTX 3050: **< 2 minutes** for the default dataset.

### Quick smoke-test (30 seconds)

Verify the entire pipeline runs before committing to a full train:

```
python train.py --smoke-test
```

### Custom options

```
python train.py --epochs 80 --batch-size 32 --lr 2e-4
```

---

## Running Sentinel

```
python main.py
```

Sentinel starts in **rule-based mode** immediately (no GPU required).  If a checkpoint exists in `checkpoints/`, it loads the neural model automatically and upgrades to **neural mode**.

### Flags

```
python main.py --checkpoint checkpoints/   # override checkpoint directory
python main.py --db data/sentinel.db       # override SQLite path
python main.py --debug                     # verbose logging
```

---

## Example commands to try

```
Open Chrome
Open Rider
Open GitHub Desktop
Open Task Manager
Search for every PDF modified today
Create a folder called Demo
Move every PNG into Images
Summarize this text file
Remember that my favourite IDE is Rider
Open my IDE
Turn the volume down
Mute
Take a screenshot
System info
Shutdown in 30 minutes
What can you do?
```

---

## Extending Sentinel

### Add a new intent

1. **`data/training_data.py`** — add 8+ example sentences to `TRAINING_DATA` with your new intent string.
2. **`core/agent/classifier.py`** — optionally add keyword rules to `rule_classify()` for instant fallback coverage.
3. **`core/agent/planner.py`** — add a `_plan_<intent>()` method that returns an `ExecutionPlan`.
4. **`core/agent/executor.py`** — wire the new tool/action string in `_dispatch()`.
5. Retrain: `python train.py`

### Replace the LLM

Swap `core/llm/model.py` with any PyTorch model that exposes:
- `.classify(input_ids, padding_mask) → logits`
- `.lm_logits(input_ids, padding_mask) → logits` *(optional for future generative use)*

The classifier, planner, executor, and UI are completely unaffected.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: pycaw` | `pip install pycaw comtypes` |
| Volume control does nothing | Pycaw needs COM; try running as administrator once |
| App won't open | Add the `.exe` path to `APP_LOOKUP` in `core/tools/apps.py` |
| CUDA out of memory | Reduce `--batch-size` to 8 or use CPU training |
| Model not loading | Run `python train.py` first |
| UI doesn't start | Ensure PyQt5 installed: `pip install PyQt5` |

---

## Roadmap

- [ ] Voice input (Whisper / local STT)
- [ ] Generative responses from the LLM (not just intent labels)
- [ ] Screen understanding (OCR + element detection)
- [ ] Plugin system for custom tools
- [ ] Larger pre-trained weights (GGUF / ONNX import)
- [ ] macOS / Linux tool layer

---

*Built entirely locally.  No API keys.  No cloud.  Your data stays on your machine.*
