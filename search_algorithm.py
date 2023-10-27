import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# nltk.download('wordnet')  # Descomentar si es la primera vez que se usa
#nltk.download('stopwords')  # Descomentar si es la primera vez que se usa

class TextComparer:
    def __init__(self, query, corpus):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.query = query
        self.corpus = corpus  # El corpus ya no se preprocesa

    def expand_query(self):
        synonyms = []
        for word in self.query.split():
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
        expanded_query = self.query + " " + ' '.join(synonyms)
        return expanded_query

    def compute_tfidf(self, expanded_query):
        text = [expanded_query] + self.corpus  # Asegurar que ambas son listas antes de concatenar
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text)
        return tfidf_matrix

    def compute_similarity(self, tfidf_matrix):
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
        return cosine_sim

    def semantic_analysis(self):
        query_embedding = self.model.encode([self.query])[0]
        corpus_embeddings = self.model.encode(self.corpus)
        semantic_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            corpus_embeddings
        )[0]
        return semantic_scores

    def evaluate_nodes(self):
        expanded_query = self.expand_query()
        tfidf_matrix = self.compute_tfidf(expanded_query)  # Definir tfidf_matrix
        cosine_sim = self.compute_similarity(tfidf_matrix)
        semantic_scores = self.semantic_analysis()
        combined_scores = np.maximum(cosine_sim*0.6, semantic_scores*0.4)
        print(combined_scores)
        return combined_scores


