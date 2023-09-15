from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
#import skops.io as sio
import json 
import os
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
def check_plagiarism(content1, content2):
    # content1 = read_file(file1_path)
    # content2 = read_file(file2_path)

    if not content1 or not content2:
        return "One or both of the files could not be read."

    vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
    similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

    vectors = vectorize([content1, content2])

    sim_score = similarity(vectors[0], vectors[1])[0][1]
    
    return sim_score


class ScoringItem(BaseModel): 
    String1: str #/ 1, // Float value  
    String2:str # "Non-Manager", # Manager or Non-Manager

#model=pickle.load(open('okmodel.pkl','rb'))

@app.post('/')
async def scoring_endpoint(item:ScoringItem): 
   # df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    #yhat = model.predict(df)
    #return {"prediction":int(yhat)}
    print(type(item))
    content1=item.String1
    content2=item.String2
    result = check_plagiarism(content1, content2)
    
    dir = {
        "similar": bool(result>.47)
    }
    return dir
