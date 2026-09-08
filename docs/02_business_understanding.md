# Business Understanding

## Stakeholder Map
- Risk — owns loss reduction; measured on fraud loss rate and capture rate; wants tighter controls
- Product — owns seller growth/onboarding conversion; measured on activation and GMV; wants low friction
- Operations — owns manual review queues; measured on review SLA and cost-per-review; wants fewer false positives
- Engineering — owns real-time enforcement systems; measured on reliability; wants low-latency, maintainable logic

These four are structurally in tension — every policy design (Day 15) must be
defensible against pushback from at least three of these four.

## Business Economics
- Revenue: take-rate on transaction volume (% of GMV)
- Loss exposure: chargebacks, refund abuse, unrecoverable payouts, buyer-protection liability
- Core asymmetry: a new seller with no track record could be a legitimate
  business or a fraud ring — almost no information exists at Day 0 to
  distinguish them, and misclassifying in either direction is costly.
