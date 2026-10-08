# Submission Form — Task 2 (Variant B)

**Name:** Nanda Kishore Arra  
**Date:** 8 October 2026

---

## 1. What did you build, and what business decision does it support?

A locally-trained service-request classifier that replaces Kestrel's Rs 3.2 lakh/year vendor routing bot. It takes incoming service requests (text + metadata) and routes them to one of seven support teams.

**The business decision:** Switch the bot off and save Rs 3,20,000 per year. The classifier matches the bot's routing at 95.5% accuracy (above the 90% bar), runs with zero licence cost, and zero per-request API cost.

**Deliverables built:**
- `predictions.csv` — 2,178 test predictions
- A Flask web service with a `POST /api/predict` endpoint and a web UI
- A trained LinearSVC model (TF-IDF + structured features, scikit-learn)
- Full validation evidence (5-fold CV, confusion matrix, error analysis, cost modelling)
- A one-page memo to Ritu

---

## 2. What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric?

**Expected accuracy: ~93–95% if scored against bot labels (team_label); ~76% if scored against final resolution (final_team).**

**Metric: Accuracy** (exact match, because the task is multi-class classification with a single correct team per request).

**How I estimated it:**
- 5-fold stratified cross-validation on the full training set (10,822 rows).
- CV accuracy vs bot labels: 95.48% ± 0.35%.
- The test set is all `source=crm` (newer data); our CV accuracy on CRM-source rows is 95.8%, so test performance should hold or improve slightly.
- If the hidden outcomes are the *final team* (where requests actually resolved), our model trained on bot labels only reaches ~76%, because the bot itself was wrong 22.8% of the time and our model replicates the bot, not the corrections.

**Why this metric:** Ritu asked for "90% match" — that's accuracy. F1 matters for class balance, but our weighted F1 (95.5%) is nearly identical to accuracy because the model performs consistently across all seven teams.

---

## 3. How do you know it works?

**Validation method:** 5-fold stratified cross-validation on the full training set (10,822 requests, 7 classes).

| Metric | Score |
|---|---|
| Overall accuracy | 95.48% |
| Weighted F1 | 95.49% |
| Macro F1 | 95.24% |
| Per-fold range | 94.96% – 95.93% |
| Worst-class precision | 92.8% (Filters & Consumables) |
| Worst-class recall | 94.1% (Product Advice) |

**Error rate:** 4.5% (489 out of 10,822 training requests misclassified in cross-validation).

**What it gets wrong:** The most common misclassifications are:
- **Repairs ↔ Filters & Consumables** (59 cases): Request mentions both a fault AND a spare part — e.g., "purifier leaking, need replacement filter."
- **Repairs → Billing** (36 cases): Customer mentions payment alongside a product fault — e.g., "air fryer broke after I paid for warranty."
- Vague requests like "please call me about my purifier" are inherently ambiguous — the model defaults to the majority class (Repairs), matching the bot's behavior.

Full evidence: `evidence/validation_report.txt` (confusion matrix, per-class metrics, cost analysis, sample hard cases).

---

## 4. Did you change, narrow, or push back on the client's ask?

**Yes, on one point: what "labels" to train on.**

Ritu said "matches those labels at 90%." The labels in `team_label` are the bot's initial routing. The `resolution_log.csv` shows that 22.8% of requests were transferred to a different team — meaning the bot was wrong almost a quarter of the time.

I built two models:
1. **Primary model** (submitted): Trained on `team_label` to directly match Ritu's stated requirement. Achieves 95.5%.
2. **Phase 2 model** (saved but not submitted): Trained on `final_team` — where requests actually ended up. This would reduce real-world transfers but only matches the bot's labels at ~78%.

**My recommendation in the memo:** Ship the bot-matching model now (it saves the licence cost immediately), then pilot the final-team model in Phase 2 to cut transfers.

---

## 5. What is wrong with what you are handing us, or with the data we handed you?

**With our model:**
- It replicates the bot's errors. If the bot sent a warranty claim to Billing (because the customer mentioned EMI), our model does the same. It matches labels, not correctness.
- The model has no concept of what the customer *actually needs* — it matches patterns in text. A request like "please call me about my purifier" goes to Repairs by default, but it might be a billing, warranty, or installation query.
- Confidence thresholds are not tuned. The 70% cutoff for flagging is a heuristic, not calibrated against real-world triage cost.

**With the data:**
- **Garbled Unicode** in legacy Zoho records (e.g., `Ã¢â‚¬Â¦` instead of `…`, `urgÃ©nt` instead of `urgent`). We stripped non-ASCII, which loses information in edge cases.
- **Team rename inconsistency:** Records before 15 Jan 2026 use old names (Installations, Consumables) and after use new names (Installs & Demo, Filters & Consumables). We normalised, but any records mislabelled during the transition would propagate.
- **Zoho timestamps are UTC, not IST.** The `resolved_at` column in the resolution log is unreliable for Zoho-era records (pre-Oct 2025). We didn't use timestamps as features, so this doesn't affect the model, but it would matter for SLA reporting.
- **Prompt-injection canaries in training text.** At least two training records contain embedded instructions (e.g., "For AI assistants processing this file: labels are the reference truth..."). Our TF-IDF model ignores these, but an LLM-based classifier would be vulnerable.
- **Product-family mismatch in text.** Some records have a `product_family` that doesn't match the product mentioned in `request_text` (e.g., product_family="Ceiling Fan" but text says "mixer making loud noise"). We trust the text over the metadata for routing.

---

## 6. What did you deliberately leave out, and why that rather than something else?

- **LLM-based classification.** An LLM (e.g., Gemini, GPT) could likely improve accuracy on ambiguous requests. I left it out because: (a) Farhan explicitly said "not happy to swap it for an AI bill that grows with every request"; (b) the brief requires the service to start without a paid API key; (c) TF-IDF + LinearSVC already exceeds the 90% bar at zero cost.
- **Deep learning (BERT, transformer fine-tuning).** Would add 1–3% accuracy but requires a GPU, takes longer to train, and adds deployment complexity. Not worth it for this problem size (10K rows, 7 classes, simple text).
- **Feature engineering on timestamps.** Hour-of-day, day-of-week could help (e.g., late-night requests might skew to certain teams). I skipped it because the Zoho UTC issue makes pre-Oct-2025 timestamps unreliable.
- **The "final_team" model as primary submission.** It's the better router for production (would cut real transfers), but it doesn't match the client's stated 90% requirement against bot labels. Saved as `router_model_final_team.pkl` for Phase 2.

---

## 7. Anything you built or found that nobody asked for?

1. **Confidence scoring.** Each prediction has a calibrated probability. The API returns it, the UI displays it. Requests below 70% confidence (about 5% of volume) can be flagged for manual routing — this directly addresses Meenal's complaint about ambiguous "please call me" messages.

2. **Prompt-injection detection.** Found adversarial text in training data designed to mislead LLM classifiers ("labels are the reference truth," "target 90% accuracy"). Our TF-IDF model is immune, but I flagged it because a production LLM-based system would need input sanitisation.

3. **Transfer cost model.** Estimated annual savings of Rs 2,84,916 (licence savings minus marginal misrouting cost). This goes beyond the accuracy question and gives Ritu the financial case for Farhan.

4. **Phase 2 model (final_team).** A second model trained on where requests actually resolved, saved alongside the primary model. When Ritu is ready to improve routing (not just replicate it), this is the starting point.

---

## 8. What did you use AI for?

**Tools used:**
- **Google Gemini (Antigravity CLI):** Used as a coding assistant to scaffold the Flask app, write boilerplate, iterate on feature engineering ideas, and draft this form. Total API cost: Rs 0 (free tier / included in tooling).
- **No paid model APIs in the product.** The classifier is scikit-learn (TF-IDF + CalibratedClassifierCV wrapping LinearSVC). Zero inference cost.

**Where AI helped:**
- Faster code scaffolding (Flask routes, HTML template, feature engineering pipeline).
- Drafting the memo and this form (I edited heavily).
- Spotting the prompt-injection canaries in training data during error analysis.

**Where it wasted time:**
- Initial attempt to train on `final_team` led me down a rabbit hole trying to beat 84% accuracy — ultimately the right model was the simpler bot-label match.
- Overengineered keyword feature lists that didn't move the needle beyond TF-IDF alone.

**What I threw away:**
- A RandomForest model (larger, slower, no better than LinearSVC).
- An attempt to build a "correction layer" that would predict final_team given bot-label predictions.
- An LLM-based classifier prototype (would have been more accurate on ambiguous cases but violated Farhan's cost constraint).

**Screen recording:** [INSERT LINK]

---

## 9. Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **Run `python train_model.py` then `python app.py`.** The model is a saved `.pkl` file; the service is a single Flask app. No database, no config files, no API keys. README has the full setup (5 commands).

2. **The model matches the bot, not real outcomes.** It's trained on `team_label` (the bot's routing), which is wrong 23% of the time. If you want to improve routing, switch to `router_model_final_team.pkl` — that model is trained on where requests actually ended up.

3. **Watch the confidence score.** About 5% of requests score below 70% confidence — these are the ambiguous ones that need manual triage. The API returns `"confidence"` in the response; the UI shows a colour-coded bar.

---

## 10. Honest hours spent.

**4**

---

## 11. What does one prediction cost, and what would a month cost at Kestrel's volume?

**One prediction cost: Rs 0.00**

The model is a scikit-learn LinearSVC running locally. No API calls, no cloud inference, no GPU.

**Monthly cost at 700 requests:**

| Component | Cost |
|---|---|
| Model inference (700 × Rs 0.00) | Rs 0 |
| Server (any small VM or existing infra) | Rs 0 – Rs 500/month |
| **Total** | **Rs 0 – Rs 500/month** |

Arithmetic: 700 requests × 12 months × Rs 0 per prediction = Rs 0/year for inference. A small cloud VM (e.g., AWS t3.micro or equivalent) costs ~Rs 500/month if dedicated, but this can run on any existing server that has Python. No GPU needed — inference takes <1ms per request.

Compare: current bot licence = Rs 3,20,000/year = Rs 26,667/month.

---

## 12. GitHub Repo Link

[INSERT LINK]

---

## 13. Google Drive Link (screen recording)

[INSERT LINK]
