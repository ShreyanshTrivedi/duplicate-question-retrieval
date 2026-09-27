# Long-query BM25 ranking review (current pipeline)

This is a **current-only** review. No pre-fix BM25 implementation or saved pre-fix rankings exist in the workspace, so an exact before/after ranking diff cannot be reproduced honestly.

The current BM25 uses light preprocessing: lowercase and punctuation removal, while retaining stopwords and unstemmed lexical forms. `HIT` denotes a qrel document.

## qid=1353 (7 whitespace words)
Query: Which book is best for Neet chemistry?
Current BM25: hits@10=1/1; P@10=0.1000; Recall@10=1.0000; MAP@10=1.0000; nDCG@10=1.0000
Qrels:
- 1354: Which is the best book for chemistry for NEET?
Current BM25 top-10:
1. [HIT] 30.1642 1354: Which is the best book for chemistry for NEET?
2. [MISS] 26.5705 2598: Which book to use for preparing chemistry for neet?
3. [MISS] 26.5674 2599: Which chemistry objective book is good for NEET 2017?
4. [MISS] 25.0357 357703: Which is the best practice book for physics and chemistry class 11 NEET?
5. [MISS] 24.1480 244384: Which is the best book to study chemistry for  the NEET & AIIMS entrance exams?
6. [MISS] 23.9916 73622: Which book is best for organic chemistry?
7. [MISS] 23.9916 188909: Which is the best book for chemistry?
8. [MISS] 22.9110 60881: Which is the best chemistry book for JEE?
9. [MISS] 22.9110 227069: Which is best chemistry book for class 11?
10. [MISS] 22.9110 368088: Which is the best book for physical chemistry?

## qid=4119 (13 whitespace words)
Query: How can I unlock an iPhone without knowing the Apple ID or password?
Current BM25: hits@10=1/1; P@10=0.1000; Recall@10=1.0000; MAP@10=1.0000; nDCG@10=1.0000
Qrels:
- 473237: I forgot my Apple ID and password. How do I unlock my iPhone 4S?
Current BM25 top-10:
1. [HIT] 32.3155 473237: I forgot my Apple ID and password. How do I unlock my iPhone 4S?
2. [MISS] 30.4961 249193: How can I recover deleted iPhone text messages, remotely, without requiring the phone or Apple ID or password?
3. [MISS] 30.4200 377788: How can you restore an Apple ID and password without iTunes?
4. [MISS] 29.4830 483659: How can I track an iPhone without iCloud using my apple id?
5. [MISS] 27.7392 307013: I forgot my Apple ID, but I know my Apple ID password. How can I know my Apple ID?
6. [MISS] 27.6190 374115: How can I find my iPhone by Apple ID without iCloud?
7. [MISS] 27.5290 233558: How can I unlock an iPod touch found without password?
8. [MISS] 27.4247 209455: How can I change my Apple ID password?
9. [MISS] 26.7439 456543: How do you Apple unlock an iPhone 4?
10. [MISS] 26.2535 479962: Can I unlock an iphone passcode without a computer?

## qid=5569 (9 whitespace words)
Query: What do Trump supporters think of his Cabinet picks?
Current BM25: hits@10=4/4; P@10=0.4000; Recall@10=1.0000; MAP@10=0.8042; nDCG@10=0.9047
Qrels:
- 5570: What do Trump supporters think about his recent cabinet appointments?
- 11964: What do Trump supporters think of Trump's prospective cabinet?
- 31889: How do Trump supporters feel about his cabinet selection?
- 115317: How do Trump supporters feel about his cabinet nominees?
Current BM25 top-10:
1. [HIT] 34.5425 5570: What do Trump supporters think about his recent cabinet appointments?
2. [MISS] 34.3271 170385: What do you think of trump's cabinet picks?
3. [HIT] 33.0062 11964: What do Trump supporters think of Trump's prospective cabinet?
4. [HIT] 30.7088 31889: How do Trump supporters feel about his cabinet selection?
5. [HIT] 30.7088 115317: How do Trump supporters feel about his cabinet nominees?
6. [MISS] 28.6020 205264: Do you support Donald Trump's cabinet picks?
7. [MISS] 28.2413 356160: In an honest assessment, what do you think of Trump’s cabinet picks?
8. [MISS] 27.4075 274966: What are Trump's most controversial cabinet picks?
9. [MISS] 26.3132 11965: To anyone who voted for President-Elect Donald Trump: What do you think about Trump's picks so far for his cabinet?
10. [MISS] 25.5953 42187: If Trump were impeached before he takes office, would his cabinet picks be invalidated?

## qid=9495 (7 whitespace words)
Query: How come ISIS has never attacked Israel?
Current BM25: hits@10=3/5; P@10=0.3000; Recall@10=0.6000; MAP@10=0.2267; nDCG@10=0.4328
Qrels:
- 9496: Why are there no ISIS attacks on Israel?
- 64158: Why does not isis attack Israel? Whatare the stakes of israel in ISIS?
- 119068: Why does ISIS never 'disturb' Israel?
- 119069: Why isn't ISIS attacking Israel?
- 153250: Why doesn't ISIS have terrorist attacks in Israel?
Current BM25 top-10:
1. [MISS] 26.6882 102160: Has ISIS ever attacked Iran?
2. [HIT] 25.3414 119068: Why does ISIS never 'disturb' Israel?
3. [MISS] 20.4381 385062: How is Israel fighting ISIS?
4. [MISS] 19.9866 9434: Is Israel funding ISIS?
5. [MISS] 19.9866 385061: Is Israel fighting ISIS?
6. [HIT] 19.3867 64158: Why does not isis attack Israel? Whatare the stakes of israel in ISIS?
7. [MISS] 18.9458 9435: Are Israel and ISIS allies?
8. [MISS] 18.9458 274496: Is israel secretly supporting ISIS?
9. [MISS] 18.3766 288689: Why hasn't ISIS attacked China or Russia?
10. [HIT] 18.0080 119069: Why isn't ISIS attacking Israel?
