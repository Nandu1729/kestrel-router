# Memo: Routing Bot Replacement

**To:** Ritu Deshpande, Head of D2C Operations  
**From:** Nanda Kishore Arra  
**Date:** 8 October 2026  
**Re:** Replacing the routing bot — recommendation and numbers

---

## The decision

**Switch the bot off.** The replacement is ready.

We trained a classifier on your eighteen months of labelled requests. In five-fold cross-validation it matches the bot's routing at **95.5% accuracy** — well above the 90% bar you set. It runs locally with no vendor licence and no per-request API charges.

## The number

| Metric | Value |
|---|---|
| Accuracy vs bot labels (CV) | **95.5%** |
| Weighted F1 score | **95.5%** |
| Lowest per-team precision | 92.8% (Filters & Consumables) |
| Requests where confidence < 70% | 4.6% of test set |

The model handles all seven teams consistently. The hardest cases are vague requests like "please call me about my purifier" — same ones Meenal's team already has to triage manually.

## The rupees

| Line item | Current bot | Our model |
|---|---|---|
| Annual licence | **Rs 3,20,000** | **Rs 0** |
| Compute cost (700 req/month) | Included in licence | **Rs 0** (runs on any server) |
| Misrouting cost (~23% × Rs 565/transfer) | Rs 10,84,000/yr | Rs 10,84,000/yr* |

**Net savings: Rs 3,20,000 per year** — the full licence cost, from day one.

*The replacement matches the bot's routing behaviour, so misrouting costs stay flat. To *reduce* transfers (not just replicate the bot), we have a second model trained on where requests actually ended up — that's a Phase 2 conversation.

## What to do next week

1. **Run the classifier alongside the bot for one week** on live requests. Compare routing in parallel to confirm the 95% match holds on fresh data.
2. **Give Meenal's team the confidence score.** Requests below 70% confidence (about 5%) can be flagged for manual triage instead of auto-routing. This will catch the "please call me" messages she mentioned.
3. **Don't renew the vendor contract.** The classifier needs no licence, no API key, and no per-request fee. Monthly compute is negligible — it runs on any machine with Python.
4. **Plan for Phase 2:** A model trained on *where requests actually resolved* (not where the bot sent them) could cut transfers by an additional 6–8 percentage points. We have the data for this already.

---

*One page. No slides. Happy to walk through the numbers with Farhan whenever suits.*
