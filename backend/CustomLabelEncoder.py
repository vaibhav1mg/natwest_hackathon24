import numpy as np
from sklearn.preprocessing import LabelEncoder

# Custom LabelEncoder to handle unseen categories
class CustomLabelEncoder(LabelEncoder):
    def fit(self, data):
        super().fit(data)
        self.classes_ = np.append(self.classes_, "Unknown")
        return self

    def transform(self, data):
        encoded = np.array([self.classes_.tolist().index(x) if x in self.classes_ else len(self.classes_) - 1 for x in data])
        return encoded
