﻿# simple_rag_2

## create a virtual environment

`python -m venv venv`

### for windows activate the virtual env using

`.\venv\Scripts\activate`

### for mac/linux

`source venv/bin/activate`

## install required python packages

`pip install -r requirements.txt`

## run a web app

`uvicorn main:app --reload`

## test out the web app online here

`https://really-uncommon-rat.ngrok-free.app/`

# api calling locally

### api status

`curl http://127.0.0.1:8000/api/status`

## sample querying

### on linux/mac

`curl -X POST http://127.0.0.1:8000/api/chat \
-H "Content-Type: application/json" \
-d '{"query": "who rejects the marriage"}'`

### on windows (powershell)

`curl -Uri http://127.0.0.1:8000/api/chat -Method POST -ContentType "application/json" -Body '{"query": "অপরিচিতা গল্পে অনুপমের মামার সঙ্গে কার তুলনা করা হয়েছে?"}'`

# Used tools

llm = gemini-2.5-flash<br>
embedding_model = Qwen/Qwen3-Embedding-0.6B<br>
vectorstore = chroma<br>

# Used libraries

langchain<br>
langchain_community<br>
google-generativeai<br>
langchain-google-genai<br>
sentence-transformers<br>
chromadb<br>
pymupdf<br>
fastapi<br>
uvicorn<br>
python-dotenv<br>
langchain_huggingface<br>

# sample queries,outputs

Question: who is kalyani's brother <br>
Answer: Sorry, I am unable to answer this question.<br>

Question: who rejects the marriage<br>
Answer: Shambhunath Sen, Kalyani's father, rejects the marriage.<br>

Question: who is kalyani<br>
Answer: Kalyani is a character in the story. Her father's name is Shambhunath Sen, who is a doctor in Kanpur. At the time of her proposed marriage, Kalyani was 16 or 17 years old. Her marriage was called off due to the groom's family's greed and lack of self-respect. After this incident, Kalyani took a vow to educate girls. She is portrayed as a strong, self-respecting woman.<br>

Question: who is kalyani's father<br>
Answer: Kalyani's father is Shambhunath Sen.<br>

Question: অনুপমের পুরো নাম কি<br>
Answer: দুঃখিত, আপনার প্রশ্নটির উত্তর আমার জানা নেই।<br>

Question: গল্পের সারাংশ বোলো<br>
Answer: "অপরাজিতা" গল্পটি অনুপম নামের এক যুবককে কেন্দ্র করে আবর্তিত হয়েছে, যে তার মামার উপর সম্পূর্ণ নির্ভরশীল এবং ব্যক্তিত্বহীন। তার বাবা উকিল হিসেবে প্রচুর অর্থ উপার্জন করলেও ভোগ করার সুযোগ পাননি। অনুপমের বয়স সাতাশ হলেও সে মায়ের কোলে গণেশের ছোট ভাইয়ের মতো। তার মামা, যিনি তার ভাগ্যের প্রধান এজেন্ট, ধনী পাত্রী পছন্দ করেন না, কিন্তু যৌতুকের প্রতি তার তীব্র আসক্তি ছিল।

মামার পছন্দ অনুযায়ী, কানপুরের ডাক্তার শম্ভুনাথ সেনের পনেরো বছর বয়সী মেয়ে কল্যাণীর সাথে অনুপমের বিয়ের কথা হয়। বিয়ের দিন মামা গহনা যাচাই করার জন্য একজন জহুরিকে সাথে নিয়ে আসেন। এই ঘটনায় কল্যাণীর বাবা শম্ভুনাথ সেন অত্যন্ত অপমানিত বোধ করেন এবং বিয়ে ভেঙে দেন। তিনি বলেন, "ঠাট্টা তো আপনিই করিয়া সারিয়াছেন।"

এই ঘটনায় অনুপম লজ্জিত ও অপমানিত হয়, কিন্তু মামার বিরুদ্ধে কোনো প্রতিবাদ করতে পারে না। পরবর্তীতে, অনুপম তীর্থযাত্রায় বেরিয়ে ট্রেনে কল্যাণীর সাথে আবার দেখা করে। কল্যাণী তখন মেয়েদের পড়ানোর কাজে নিজেকে নিয়োজিত করেছে এবং দেশের সেবায় ব্রতী হয়েছে। সে আর বিয়ে করেনি। অনুপম কল্যাণীর আত্মমর্যাদা ও স্বাধীনচেতা মনোভাব দেখে মুগ্ধ হয় এবং তার প্রতি এক গভীর আকর্ষণ অনুভব করে। গল্পটি যৌতুক প্রথার কুফল এবং নারীর আত্মমর্যাদা ও স্বাধীন ব্যক্তিত্বের জয়কে তুলে ধরেছে। অনুপম তার নিষ্ক্রিয় জীবনের ব্যর্থতা উপলব্ধি করে এবং কল্যাণীর সান্নিধ্য পাওয়ার আকাঙ্ক্ষা নিয়েই তার জীবন কাটে।<br>

Question: what is the capital of Bangladesh<br>
Answer: Sorry, I am unable to answer this question.<br>

Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?<br>
Answer: কল্যাণীর বিয়ের সময় তার প্রকৃত বয়স ছিল ১৫ বছর।<br>

Question: who is kalyani's father<br>
Answer: Kalyani's father is Shambhunath Sen.<br>

Question: who are the main characters of the story<br>
Answer: The main characters of the story 'Aparichita' are Anupam (the narrator), Kalyani, Anupam's uncle, and Shambhunath Sen (Kalyani's father).<br>

Question: do kalyani and onupom finally get married<br>
Answer: No, Kalyani and Anupam do not get married in the story. The marriage is called off by Kalyani's father due to the groom's uncle's behavior regarding the dowry. Kalyani herself later states that she will not marry.<br>

Question: is it a sad story, or what is the story type of it... what could be tagline of this story<br>
Answer: The story about Kalyani and Anupam, titled "Aparichita," is a prose story, specifically a confessional narrative told from the perspective of the protagonist, Anupam.

Yes, it is considered a sad story, particularly for Anupam, who lives with regret and mental anguish due to his inability to act and the broken marriage proposal.

A suitable tagline for the story could be: "A tale of unfulfilled love and the anguish of a lost opportunity."<br>
