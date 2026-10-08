# Submission Form — Task 2 (Variant B)

**Name:** Nanda Kishore Arra  
**Date:** 8 October 2026

---

## 1. What did you build, and what business decision does it support? State the number and the rupees.

A self-contained service-request classifier (TF-IDF + calibrated LinearSVC) and Flask web service that automatically routes incoming queries across Kestrel's 7 teams. It supports terminating the vendor routing bot contract to save Rs 3,20,000 annually. The model matches historical bot routing at 95.5% accuracy (exceeding Ritu's 90% target) at Rs 0 per-request cost.

---

## 2. What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it. We compare this with the real score.

Expected accuracy: ~94–95% if evaluated against bot labels (team_label); ~76% if evaluated against final resolutions (final_team). Estimated via 5-fold stratified cross-validation on 10,822 training rows (95.48% ± 0.35%). We use Accuracy because Kestrel's brief explicitly required a "90% match" against existing labels, and multi-class weighted F1 is equally high (95.49%).

---

## 3. How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.

Validated via 5-fold stratified CV on all 10,822 training requests. It achieved 95.48% accuracy and 95.49% weighted F1, with an error rate of 4.5% (489/10,822). Errors concentrate where requests combine faults with spare parts (Repairs vs Filters & Consumables, 59 cases) or mention payments alongside broken items (Repairs vs Billing, 36 cases).

---

## 4. Did you change, narrow, or push back on the client's ask? What, when, and why. [can only raise your score]

Yes. Resolution logs revealed the vendor bot misrouted 22.8% of past tickets. I trained two models: a primary model on team_label to meet Ritu's immediate 90% benchmark to drop the Rs 3.2L license (95.5% accuracy), and a secondary model on final_team (true resolutions) saved in the repo for Phase 2 to systematically reduce agent transfer costs.

---

## 5. What is wrong with what you are handing us, or with the data we handed you? Be specific: bugs, shortcuts, columns you did not trust, rows that looked wrong. [can only raise your score]

Our model replicates the bot's historical misroutes on payment/fault overlaps and defaults vague queries to majority classes. In the data: legacy Zoho records had garbled Unicode and unconverted UTC timestamps (policy §9); team renames created 9 label variants; and training texts contained adversarial prompt-injection canaries (which our TF-IDF model safely ignored).

---

## 6. What did you deliberately leave out, and why that rather than something else?

Left out paid LLMs/APIs because Finance (Farhan) vetoed recurring per-request bills and the brief required zero paid keys. Left out heavy deep learning (BERT) because it added GPU overhead without accuracy gain. Left out timestamp features due to unconverted Zoho UTC records.

---

## 7. Anything you built or found that nobody asked for?

1) Calibrated confidence scoring with a 70% threshold flagging the ~4.6% ambiguous requests for manual triage (solving Meenal's "please call me" issue). 2) Discovered adversarial prompt injections in the training set. 3) Financial transfer cost model quantifying Rs 2,84,916 net annual savings.

---

## 8. What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

Used Google Gemini (Antigravity CLI) as a coding assistant for scaffolding the Flask app, UI, and pipeline (Rs 0 cost). The production model uses zero paid APIs (scikit-learn). AI sped up boilerplate and spotted adversarial data canaries, but initial AI prompts overengineered complex neural nets that were thrown away for LinearSVC. Screen recording: [PASTE GOOGLE DRIVE LINK]

---

## 9. Your Public Google Drive Link

[PASTE YOUR GOOGLE DRIVE LINK]

---

## 10. Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. Run 'pip install -r requirements.txt' and 'PORT=8080 python app.py'—pre-trained model is in models/, runs instantly without data files or API keys.
2. The deployed model replicates historical bot routing (95.5%); to optimize real-world resolutions, switch to models/router_model_final_team.pkl.
3. Flag requests with <70% confidence for manual human triage.

---

## 11. Honest hours spent. One number.

4

---

## 12. Github Repo Link

https://github.com/Nandu1729/kestrel-router

---

## 13. What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)? Show the arithmetic. If you used no paid calls, say so.

One prediction: Rs 0.00. Monthly cost at 700 requests: Rs 0.00 (runs locally on CPU in <1ms without paid APIs). Annual savings: Rs 3,20,000 by eliminating the Rs 26,667/month vendor license.
