# Answer-quality eval — en_native, 300 validation queries
## Outcome breakdown
| outcome | count |
|---|---|
| answered | 300 |

## Answer quality (among ANSWERED queries)
| metric | value |
|---|---|
| mean word-overlap F1 vs gold answer | 0.374 |
| median word-overlap F1 vs gold answer | 0.275 |
| % with zero word overlap vs gold answer | 14.0% |
| % answer sourced from a gold-relevant passage | 45.0% |

## 10 lowest-F1 examples (for manual inspection)

**query** (PERSON): who played the wife in bernie mac show
- gold answer: Wanda
- pipeline answer: Bernie Mac with wife Rhonda in 2004.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): is it good idea to buy house
- gold answer: Yes
- pipeline answer: However, just like marriage I would argue that buying a house today is also a bad idea.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): what structure is found on the rough er
- gold answer: Ribosomes
- pipeline answer: Rough ER is found throughout the cell but the density is higher near the nucleus and the Golgi apparatus.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): what is a chemical property of selenium
- gold answer: Se
- pipeline answer: For example, selenium is a semiconductor.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): what math courses do nurses need
- gold answer: Nurses need to Translate Arabic numbers to Roman numerals, Translate Roman numerals to Arabic numbers.
- pipeline answer: (Math 1342) as the mathematics requirement for nursing bachelor’s degree programs in Texas.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): does crestor cause a cough
- gold answer: Yes
- pipeline answer: I have been dealing with a cough most of that time.
- f1=0.000  sourced_from_gold_passage=False

**query** (NUMERIC): what year did facebook ipo
- gold answer: 2012
- pipeline answer: For its part, Facebook said in its IPO documents last month that it expected the deal to close in the second quarter.
- f1=0.000  sourced_from_gold_passage=True

**query** (DESCRIPTION): is it true a summons is a call to appear in court
- gold answer: Yes
- pipeline answer: The summons announces a date by which the person or persons summoned must either appear in court, or respond in writing to the court or the opposing party or parties.
- f1=0.000  sourced_from_gold_passage=True

**query** (DESCRIPTION): kids definition of multiples
- gold answer: A multiple is the result of multiplying a number by an integer.
- pipeline answer: Multiples definition for kids.
- f1=0.000  sourced_from_gold_passage=False

**query** (NUMERIC): model of glucose and how many atp
- gold answer: 38
- pipeline answer: How many ATP molecules are produced for each molecule of glucose in glycolysis?
- f1=0.000  sourced_from_gold_passage=False
