import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
#nltk.download('omw-1.4')


# nltk.download('wordnet')  # Descomentar si es la primera vez que se usa
nltk.download('stopwords')  # Descomentar si es la primera vez que se usa

class TextComparer:
    def __init__(self, query, corpus):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.query = query
        self.corpus = corpus  # El corpus ya no se preprocesa
       # self.query = self.preprocess(query)
       # self.corpus = [self.preprocess(doc) for doc in corpus]  # Preprocesar cada documento en el corpus
    #
    # def preprocess(self, text):
    #     # Convertir el texto a minúsculas
    #     text = text.lower()
    #     # Eliminar signos de puntuación
    #     text = text.translate(str.maketrans('', '', string.punctuation))
    #     # Tokenización por palabras
    #     words = nltk.word_tokenize(text)
    #     # Eliminar stopwords
    #     stop_words = set(stopwords.words('spanish'))
    #     words = [word for word in words if word not in stop_words]
    #     # Unir palabras en un texto nuevamente
    #     text = ' '.join(words)
    #     return text





# def expand_query(self):
    #     synonyms = set()
    #     for word in self.query.split():
    #         for syn in wn.synsets(word, lang='spa'):
    #             for lemma in syn.lemmas('spa'):
    #                 synonyms.add(lemma.name())
    #
    #     expanded_query = self.query + " " + ' '.join(synonyms)
    #     print(expanded_query)
    #     return expanded_query


    def compute_tfidf(self, query):
        text = [query] + self.corpus  # Asegurar que ambas son listas antes de concatenar
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text)
        return tfidf_matrix

    def compute_similarity(self, tfidf_matrix):
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
        return cosine_sim

    def semantic_analysis(self):
        query_embedding = self.model.encode([self.query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        corpus_embeddings = self.model.encode(self.corpus)
        semantic_scores = []
        for i in corpus_embeddings:
            # Asegúrate de que 'i' también sea una matriz 2D
            i_reshaped = i.reshape(1, -1)
            elem = cosine_similarity(query_embedding, i_reshaped)[0]
            semantic_scores.append(elem)

        return semantic_scores


    def evaluate_nodes(self):
        #expanded_query = self.expand_query()
        tfidf_matrix = self.compute_tfidf(self.query)  # Definir tfidf_matrix
        cosine_sim = self.compute_similarity(tfidf_matrix)
        semantic_scores = self.semantic_analysis()
        # Asegúrate de que ambos sean arrays NumPy antes de combinarlos
        cosine_sim_array = np.array(cosine_sim).flatten()  # Aplanar el array si es necesario
        semantic_scores_array = np.array(semantic_scores).flatten()  # Convertir la lista de listas a un array NumPy
        combined_scores = np.maximum(cosine_sim_array*0.6, semantic_scores_array*0.4)
        print(combined_scores)
        return combined_scores




# # Query
# query = "¿Cuáles son los beneficios de la meditación?"
#
# # Corpus
# corpus = [
#     "La meditación es una práctica increíble que puede reducir significativamente los niveles de estrés diario.",
#     "Al meditar regularmente, he notado una mejora considerable en mi concentración y atención, lo cual es beneficioso.",
#     "Desde que comencé a meditar, he experimentado un equilibrio emocional mucho más estable y sereno.",
#     "La meditación ha desbloqueado niveles de creatividad y claridad mental que no sabía que tenía en mi interior.",
#     "Muchas personas, incluyéndome, hemos encontrado en la meditación una herramienta eficaz para regular el sueño y descansar mejor por las noches.",
#     "Se ha demostrado que la práctica de la meditación puede disminuir la presión arterial, lo que es un gran beneficio para la salud cardiovascular.",
#     "Al dedicar tiempo a la meditación, he logrado una mayor conexión y entendimiento con mi ser interior.",
#     "Estudios han mostrado que la meditación mejora la memoria y la capacidad cognitiva, algo que puedo corroborar con mi experiencia personal.",
#     "La paciencia y la tolerancia son dos virtudes que se han potenciado en mi vida gracias a la meditación regular.",
#     "En tiempos de alta ansiedad, he encontrado en la meditación una aliada efectiva para gestionar y reducir esos niveles de ansiedad."
# ]
#
# # Instanciar la clase con la consulta y el corpus
# comparer = TextComparer(query, corpus)
#
# # Evaluar nodos
# comparer.evaluate_nodes()


