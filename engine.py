import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class FairnessEngine:
    def __init__(self, data):
        self.data = data
        self.model = self._train_baseline()

    def _train_baseline(self):
        # Simplest version: Gender (0/1), Age, Education
        X = self.data[['age', 'gender_binary', 'education_num']]
        y = self.data['income_high']
        model = RandomForestClassifier()
        model.fit(X, y)
        return model

    def get_counterfactual(self, input_data):
        # input_data is a dict: {'age': 30, 'gender_binary': 0, 'education_num': 12}
        original_pred = self.model.predict_proba(pd.DataFrame([input_data]))[0][1]
        
        # Flip the gender (0 to 1 or 1 to 0)
        cf_data = input_data.copy()
        cf_data['gender_binary'] = 1 if input_data['gender_binary'] == 0 else 0
        
        cf_pred = self.model.predict_proba(pd.DataFrame([cf_data]))[0][1]
        
        return original_pred, cf_pred
