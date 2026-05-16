# importing frameworks and libraries
import faiss
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal,List
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# Loading the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


data = []

# loading the json file
with open("shl_product_catalog.json","r",encoding = "utf-8",errors = "ignore") as shl:
    content = json.loads(shl.read(),strict = False)


# keeping the needy data from the json file removing the messiness
for entry in content:
    entries = f"{entry['name']} {entry['description']} {entry['keys']} {entry['job_levels']}"
    data.append(entries)


# creating the embeddings
embeddings = model.encode(data)
embeddings = np.array(embeddings,dtype="float32")
dimension = embeddings.shape[1]  # embedding is n rows X m columns vector where n represents the no of texts,chunks and m represents the each vector dimension (text,chunk)
index = faiss.IndexFlatL2(dimension)  ## IndexFlatL2 --> Uses L2 distance (Euclidean distance) for similarity vector
index.add(embeddings)



# craeting FastAPI app
app = FastAPI()



# creating and defining llm model
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0.1
)


# checking the fastapi service whether ok
@app.get("/health")
def health():
    return {"status":"ok"}
# This endpoint is used to check whether the FastAPI service is running properly.


Role = Literal["user","system","assistant"]


# Defining the output type using pydantic BaseModel
class message(BaseModel):
    role:Role
    content:str

class ChatMessage(BaseModel):
    model:str = "gemini-2.5-flash"
    messages : List[message]

class recommendation(BaseModel):
    name:str
    test_type:str
    url:str

class ChatResponse(BaseModel):
    reply:str
    recommendations : List[recommendation]
    end_of_conversation : bool



# using fastapi post request
@app.post("/chat",response_model = ChatResponse)
def chat(request:ChatMessage,k:int = 10):
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    user_message = " ".join([m.content for m in request.messages if m.role == "user"])


    q_query = model.encode([user_message])
    q_query = np.array(q_query,dtype="float32")
    distances,indices = index.search(q_query,k)
    # indices shows the arrays of the position of the vector which is near to the query vector and distance shows the array of the distance between the positon sof the query vector and the similar vector 
    # the position which indices returns on the vector db search is same as the posiiton in the json contet of any data
    
    
    
    retrieved = []

    for i in indices[0]:  #Indices return double list so indices[0]-List of matching vectors
        if i != -1:  #-1  means no resukt found 
            retrieved.append(content[i])


    # Build catalog context fro the llm model
    catalog = "\n".join([f"{r['name']} | type : {r['keys']} | levels : {r['job_levels']} | url : {r['link']}" for r in retrieved])
    # We made catalog again because we cant give the embedding because it is in vector form amd llm do not understand and the json file we dont give because the json file hs large data and we need to give the llm only the matching vector data so we give the retrieved data
    # build conversation history
    history = "\n".join([f"{m.role}:{m.content}" for m in request.messages])

    # System prompt 
    system_prompt = f"""You are an SHL assessment recommender. Only recommend assessments from the catalog below.
    You must respond in this exact JSON format with no markdown or code fences:
    {{
    "reply": "your conversational reply",
    "recommendations": [{{"name": "...", "url": "...", "test_type": "..."}}],
    "end_of_conversation": false
    }}

    Rules:
    - If the query is vague, ask ONE clarifying question. Set recommendations to [].
    - Only recommend when you have enough context (role, level, skill needed).
    - Never invent URLs. Only use URLs from the catalog below.
    - Refuse anything not related to SHL assessments.
    - end_of_conversation is true only when you have given a final shortlist.
    - Recommend between 1 and 10 assessments when you have enough context.

    Catalog (use only these):
    {catalog}

    Conversation:
    {history}"""


# invoking the system prompt into the llm model
    response = llm.invoke(system_prompt)
    raw = response.content.strip()

    try: 
        # checking whether the raw starts with ``` if yes then removing it 
        if raw.startswith("```"):
            raw = raw.split("```")[1]    #Takes the middle content[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())   # json.loads() converts the string into object/json from string and strip means removing extra spaces and lines
    except Exception:
        return ChatResponse(
            reply = raw,
            recommendations = [],
            end_of_conversation = False
        )
    

    # returning structures response
    recs=[
        recommendation(
            name=r.get("name",""),
            test_type= r.get("test_type",""),
            url=r.get("url","")
        )
        for r in parsed.get("recommendations",[])
    ]

    
    return ChatResponse(
        reply = parsed.get("reply",""),
        recommendations = recs,
        end_of_conversation = parsed.get("end_of_conversation",False)
    )