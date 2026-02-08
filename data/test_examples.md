# Topic Classifier - Test Examples

Testing suite for validating the article topic classification model.

---

## 1️⃣ Business (Clear Case)

**Title:** US stocks rise as inflation data boosts investor confidence

**Body:**
```
US equity markets climbed on Tuesday after fresh inflation data suggested
price pressures may be easing. Analysts said the Federal Reserve could slow
the pace of interest rate hikes, lifting investor sentiment across sectors.
```

**Expected Output:**
```json
{
  "topic": "Business",
  "confidence": 0.9
}
```

**Rationale:** Key signals: markets, inflation, investors, Fed → strong business domain vocabulary

---

## 2️⃣ Sports (Very Strong Signal)

**Title:** Manchester City secure late victory in Champions League clash

**Body:**
```
Manchester City scored a dramatic late goal to defeat Bayern Munich
in the Champions League quarter-final. The match saw intense midfield
battles and standout performances from both goalkeepers.
```

**Expected Output:**
```json
{
  "topic": "Sports",
  "confidence": 0.95
}
```

**Rationale:** Highly distinctive vocabulary: match, goal, Champions League → sports domain

---

## 3️⃣ World / Politics (Clear Geopolitical Case)

**Title:** European leaders meet to discuss Ukraine peace efforts

**Body:**
```
Leaders from several European nations gathered in Brussels to discuss
ongoing diplomatic efforts aimed at ending the conflict in Ukraine.
The talks focused on security guarantees and humanitarian aid.
```

**Expected Output:**
```json
{
  "topic": "World",
  "confidence": 0.85
}
```

**Rationale:** Key signals: leaders, nations, conflict, diplomacy → world affairs domain

---

## 4️⃣ Sci/Tech (Technology Focus)

**Title:** New AI chip promises faster training for deep learning models

**Body:**
```
A leading semiconductor company unveiled a new AI-focused chip designed
to accelerate the training of deep learning models while reducing power
consumption. Experts say it could reshape data center architectures.
```

**Expected Output:**
```json
{
  "topic": "Sci/Tech",
  "confidence": 0.90
}
```

**Rationale:** Technical domain signals: AI, chip, deep learning, semiconductor → sci/tech

---

## 5️⃣ Ambiguous Case: Business vs Sci/Tech ⚠️

**Title:** Tech firms report strong quarterly earnings

**Body:**
```
Several major technology companies reported better-than-expected earnings
this quarter, driven by strong cloud computing demand and cost-cutting
measures across divisions.
```

**Expected Output:**
```json
{
  "topic": "Business or Sci/Tech",
  "confidence": 0.60
}
```

**Rationale & Testing Importance:**
- This is a realistic ambiguity case
- Confidence should be lower due to mixed signals (financial + technical)
- Tests threshold behavior and need for human review
- **If your model handles this correctly → behaving as expected**

---

## 6️⃣ Weak / Noisy Input (Edge Case)

**Title:** Company announces update

**Body:**
```
The company announced an update on Monday. More details are expected later.
```

**Expected Output:**
```json
{
  "topic": "Any / Uncertain",
  "confidence": 0.40
}
```

**Rationale:** Minimal context, generic language → low confidence signal. Should trigger uncertainty handling.

---

## Testing Guidelines

| Test Case | Purpose | Success Criteria |
|-----------|---------|------------------|
| 1 - Business | Clear single classification | High confidence (>0.85) |
| 2 - Sports | Strong domain signals | Very high confidence (>0.90) |
| 3 - World | Geopolitical vocabulary | High confidence (0.85-0.95) |
| 4 - Sci/Tech | Technical domain | High confidence (0.85-0.95) |
| 5 - Ambiguous | Edge case handling | Medium-low confidence (0.55-0.75) |
| 6 - Noisy | Robustness to poor input | Low confidence (<0.60) |

