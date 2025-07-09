from pydantic import BaseModel

# By default fastapi doesn't know how to interpret the input query.
# We have to explicitly inform to it, we are passing as json.
# When it is pydantic model, the output will be in json.
# In curl we also mention where to look
'''
curl -X POST http://127.0.0.1:8000/predict \                                                                                                                         (base) 
              -H "Content-Type: application/json" \
              -d '{
            "Age": 54,
            "Sex": "M",
            "ChestPainType": "ATA",
            "RestingBP": 140,
            "Cholesterol": 239,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 160,
            "ExerciseAngina": "N",
            "Oldpeak": 1.2,
            "ST_Slope": "Up"
          }'
'''

# We can also go to "http://127.0.0.1:8000/docs#/default/predict_predict_post"

class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str
