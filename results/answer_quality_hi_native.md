# Answer-quality eval — hi_native, 300 validation queries
## Outcome breakdown
| outcome | count |
|---|---|
| answered | 297 |
| insufficient_evidence | 3 |

## Answer quality (among ANSWERED queries)
| metric | value |
|---|---|
| mean word-overlap F1 vs gold answer | 0.310 |
| median word-overlap F1 vs gold answer | 0.222 |
| % with zero word overlap vs gold answer | 13.5% |
| % answer sourced from a gold-relevant passage | 32.7% |

## Real answer status among ANSWERED rows (30-row LLM-verified sample — this is what 'answered' actually means)
The pipeline's ANSWERED outcome only means a candidate extractive answer passed cheap heuristic gates — it is not a correctness verdict. This table applies the async four-dimension LLM judge (faithfulness/answerability/completeness/relevance) to a sample of those rows and reports what they actually resolve to.
| final status | count | % of sample |
|---|---|---|
| not_answered | 18 | 60.0% |
| answered | 12 | 40.0% |

## 10 lowest-F1 examples (for manual inspection)

**query** (NUMERIC): क्या ए.एन.जेड.ए.सी. दिवस है
- gold answer: ऑस्ट्रेलिया और न्यूजीलैंड में स्मरण का एक राष्ट्रीय दिवस।
- pipeline answer: ए.एन.जेड.ए.सी.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): क्या गैस प्रोग्राम है
- gold answer: जॉर्जिया के राज्य ने शिक्षक प्रमाणन मूल्यांकन कार्यक्रम को मंजूरी दे दी।
- pipeline answer: यदि गैस कैप समस्या का समाधान नहीं करता है, तो गैस में थोड़ा एस.ई.ए.एम.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): सेलेनियम की रासायनिक संपत्ति क्या है
- gold answer: से
- pipeline answer: सेलेनियम एक रासायनिक तत्व है जिसका प्रतीक Se और परमाणु संख्या 34 है।
- f1=0.000  sourced_from_gold_passage=True

**query** (DESCRIPTION): यू.एस.एस.आर. का क्या अर्थ है
- gold answer: सोवियत समाजवादी गणराज्यों का संघ
- pipeline answer: यू.एस.एस.आर.
- f1=0.000  sourced_from_gold_passage=False

**query** (ENTITY): बीट्रिक्स नाव की तरह
- gold answer: मोटर नौका बीट्रिक्स (पेंदा 163) एक उत्कृष्ट 50 मीटर विस्थापन नौका है, जिसे इतालवी पोत-कारखाने, कैंटिएरी नवाली डी टर्मोली द्वारा निर्मित किया गया है।
- pipeline answer: इसी तरह की नौकाएँ।
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): क्या यह सच है कि समन का अर्थ है अदालत में पेश होने के लिए बुलाया जाना?
- gold answer: सहमत हैं तो
- pipeline answer: आम तौर पर आपको व्यक्तिगत रूप से या प्रमाणित या पंजीकृत मेल द्वारा न्यायालय में वैध होने के लिए समन प्राप्त होना चाहिए।
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): एक्सपीडिया कार्ड के लिए आपको किस तरह का क्रेडिट स्कोर मंजूरी दी जानी चाहिए
- gold answer: 660 या उससे बेहतर
- pipeline answer: सिटी प्रीमियर पास/एक्सपीडिया कार्ड के लिए अनुमोदित होने के लिए, आपका क्रेडिट अच्छा होना चाहिए।
- f1=0.000  sourced_from_gold_passage=True

**query** (NUMERIC): ओक्लाहोमा में सबसे अधिक तापमान दर्ज किया गया तापमान क्या है
- gold answer: 120 डिग्री फारेनहाइट
- pipeline answer: ओक्लाहोमा में अब तक का सबसे ठंडा तापमान आज दर्ज किया गया।
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): परिभाषा ज्यामिति
- gold answer: अनुवाद एक ऐसा कार्य है जो एक निर्दिष्ट दिशा में हर बिंदु को एक निरंतर दूरी पर ले जाता है।
- pipeline answer: ज्यामिति परिभाषा पाठ्यपुस्तक-पाठ-परिभाषा-निर्माण-प्रतिपाद्य-निष्कर्ष-इंटरनेट गतिविधियाँ और संसाधन।
- f1=0.000  sourced_from_gold_passage=False

**query** (LOCATION): टेलिको मैदान टेनेसी कहाँ है
- gold answer: मोनरो काउंटी में।
- pipeline answer: टेलिको प्लेन्स 35°22′0″N 84°17′56″W / 35.36667°N 84.29889°W / 35.36667; -84.29889 (35.366575, -84.298955) पर स्थित है।
- f1=0.000  sourced_from_gold_passage=False
