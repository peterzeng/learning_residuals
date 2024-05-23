from gram2vec import vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class ExplainableModule():
    def __init__(self) -> None:
        pass

    def get_vector_and_score(self, doc1, doc2):
        pass

class Gram2VecModule(ExplainableModule):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        if not os.path.exists(f"{dataset}_vector_map.pkl"):
            self.cache = {}
        else:
            import pickle
            with open(f"{dataset}_vector_map.pkl", "rb") as f:
                self.cache = pickle.load(f)

        self.vectorizer_configs = {
            "pos_unigrams":1,
            "pos_bigrams":1,
            "func_words":1,
            "punctuation":1,
            "letters":1,
            "emojis":1,
            "dep_labels":1,
            "morph_tags":1,
            "sentences":1
        }

    def cache_features(self, u_id, vector): 
        self.cache[u_id] = vector
    
    def get_vector(self, doc, uid):
        if uid in self.cache:
            return self.cache[uid]
        else:
            vector = vectorizer.from_documents([doc], config = self.vectorizer_configs).values
            self.cache_features(uid, vector)
            return vector

    def get_vector_and_score(self, doc1, doc2, uid1='', uid2=''):

        if uid1 in self.cache: # This is for during TA2 eval when we have computed the query vector already
            vector_1 = self.cache[uid1]
        else:
            vector_1 = vectorizer.from_documents([doc1], config = self.vectorizer_configs).values
            self.cache_features(uid1, vector_1)

        if uid2 in self.cache:
            vector_2 = self.cache[uid2]
        else:
            vector_2 = vectorizer.from_documents([doc2], config = self.vectorizer_configs).values
            self.cache_features(uid2, vector_2)

        cosine_sim = cosine_similarity(vector_1, vector_2)[0][0]
        return cosine_sim

    def save_cache(self):
        import pickle
        if self.dataset == "reddit":
            with open("reddit_vector_map.pkl", "wb") as f:
                pickle.dump(self.cache, f)
        elif self.dataset == "fanfiction":
            with open("fanfiction_vector_map.pkl", "wb") as f:
                pickle.dump(self.cache, f)

