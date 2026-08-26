# Answer-quality eval — en_native, 300 validation queries
## Outcome breakdown
| outcome | count |
|---|---|
| answered | 274 |
| insufficient_evidence | 26 |

## Answer quality (among ANSWERED queries)
| metric | value |
|---|---|
| mean word-overlap F1 vs gold answer | 0.390 |
| median word-overlap F1 vs gold answer | 0.279 |
| % with zero word overlap vs gold answer | 10.9% |
| % answer sourced from a gold-relevant passage | 46.4% |

## Real answer status among ANSWERED rows (30-row LLM-verified sample — this is what 'answered' actually means)
The pipeline's ANSWERED outcome only means a candidate extractive answer passed cheap heuristic gates — it is not a correctness verdict. This table applies the async four-dimension LLM judge (faithfulness/answerability/completeness/relevance) to a sample of those rows and reports what they actually resolve to.
| final status | count | % of sample |
|---|---|---|
| not_answered | 22 | 73.3% |
| answered | 8 | 26.7% |

## 10 lowest-F1 examples (for manual inspection)

**query** (NUMERIC): what age did martin luther king died
- gold answer: 51
- pipeline answer: Martin Luther King, Jr.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): translation definition geometry
- gold answer: Ttranslation is a function that moves every point a constant distance in a specified direction.
- pipeline answer: Geometry Definition: Translation.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): what is a chemical property of selenium
- gold answer: Se
- pipeline answer: For example, selenium is a semiconductor.
- f1=0.000  sourced_from_gold_passage=False

**query** (ENTITY): what vitamins to take for hair loss in women
- gold answer: B12,vitamin C
- pipeline answer: Hair Loss in women can be a source of much despair.
- f1=0.000  sourced_from_gold_passage=False

**query** (NUMERIC): what year did facebook ipo
- gold answer: 2012
- pipeline answer: The company sent an amended IPO filing to the SEC on May 3.
- f1=0.000  sourced_from_gold_passage=True

**query** (ENTITY): what vegetables can be grilled
- gold answer: Onions,flat mushrooms
- pipeline answer: How To Grill Vegetables For Amazing Flavor.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): what causes a massive stroke
- gold answer: Decreased blood supply to the brain, blood clot that forms inside the heart.Infections, trauma, heart disorders, sickle cell disease and dehydration.
- pipeline answer: Cardiogenic causes account for more stroke in young adults.
- f1=0.000  sourced_from_gold_passage=True

**query** (NUMERIC): average annual rainfall for young harris ga
- gold answer: 54 inches
- pipeline answer: On average, there are 205 sunny days per year in Young Harris, Georgia.
- f1=0.000  sourced_from_gold_passage=True

**query** (DESCRIPTION): what structure is found on the rough er
- gold answer: Ribosomes
- pipeline answer: Rough ER is found throughout the cell but the density is higher near the nucleus and the Golgi apparatus.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): what math courses do nurses need
- gold answer: Nurses need to Translate Arabic numbers to Roman numerals, Translate Roman numerals to Arabic numbers.
- pipeline answer: (Math 1342) as the mathematics requirement for nursing bachelor’s degree programs in Texas.
- f1=0.000  sourced_from_gold_passage=False
