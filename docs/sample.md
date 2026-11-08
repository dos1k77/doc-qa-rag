# What is RAG?

Retrieval-Augmented Generation (RAG) is a technique where a language model
answers a question using passages found in your own documents. First, the
documents are split into small chunks. Then a search step finds the chunks most
similar to the question. Finally, those chunks are given to the model as
context, so the answer is grounded in real text instead of memory alone.

Common ranking methods are BM25 (keyword based) and embeddings (meaning based).
BM25 is simple, fast and works well when the question shares words with the
document.

# RAG деген не?

RAG — тілдік модель сұраққа өз құжаттарыңыздан табылған үзінділер негізінде
жауап беретін әдіс. Алдымен құжаттар шағын бөліктерге бөлінеді. Содан кейін
іздеу қадамы сұраққа ең ұқсас бөліктерді табады. Соңында бұл бөліктер модельге
контекст ретінде беріледі. Сондықтан жауап тек модельдің жадына емес, нақты
мәтінге сүйенеді.
