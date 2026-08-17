# Answer-quality eval — hi_native, 300 validation queries
## Outcome breakdown
| outcome | count |
|---|---|
| answered | 300 |

## Answer quality (among ANSWERED queries)
| metric | value |
|---|---|
| mean word-overlap F1 vs gold answer | 0.310 |
| median word-overlap F1 vs gold answer | 0.222 |
| % with zero word overlap vs gold answer | 13.3% |
| % answer sourced from a gold-relevant passage | 32.7% |

## 10 lowest-F1 examples (for manual inspection)

**query** (NUMERIC): क्या ए.एन.जेड.ए.सी. दिवस है
- gold answer: ऑस्ट्रेलिया और न्यूजीलैंड में स्मरण का एक राष्ट्रीय दिवस।
- pipeline answer: ए.एन.जेड.ए.सी.
- f1=0.000  sourced_from_gold_passage=False

**query** (NUMERIC): मार्टिन लूथर किंग की मृत्यु किस उम्र में हुई थी
- gold answer: ५१
- pipeline answer: मार्टिन लूथर किंग, जूनियर एक अमेरिकी पादरी और नागरिक अधिकार नेता थे जिन्हें 4 अप्रैल 1968 के गुरुवार को 39 वर्ष की आयु में टेनेसी के मेम्फिस में लोरेन मोटल में घातक रूप से गोली मार दी गई थी।
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): परमाणु बम गिराने से कैसे जान बच गई?
- gold answer: इससे युद्ध समाप्त हो गया।
- pipeline answer: यह कहना असंभव है कि जापान के परमाणु बम विस्फोट से कितनी जानें बचाई गईं क्योंकि किसी को भी पता नहीं है कि अगर बम नहीं गिराए गए होते तो जापान कब आत्मसमर्पण कर लेता।
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): क्या गैस प्रोग्राम है
- gold answer: जॉर्जिया के राज्य ने शिक्षक प्रमाणन मूल्यांकन कार्यक्रम को मंजूरी दे दी।
- pipeline answer: यदि गैस कैप समस्या का समाधान नहीं करता है, तो गैस में थोड़ा एस.ई.ए.एम.
- f1=0.000  sourced_from_gold_passage=False

**query** (NUMERIC): ग्लूकोज का मॉडल और कितने ए.टी.पी.
- gold answer: 38
- pipeline answer: ग्लाइकोलाइसिस में ग्लूकोज के प्रत्येक अणु के लिए कितने ए.टी.पी.
- f1=0.000  sourced_from_gold_passage=False

**query** (DESCRIPTION): सी-प्रतिक्रियाशील प्रोटीन का क्या अर्थ है
- gold answer: शोथ का एक संवेदनशील चिह्नक।
- pipeline answer: यह पृष्ठ सी-प्रतिक्रियाशील प्रोटीन शब्द के सभी संभावित अर्थ और अनुवाद प्रदान करता है।
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
