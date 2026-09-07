import json
api_request={
 "model1":"chat_model",
 "message":[
     {"role":"user","content":"explain API"}
 ],
 "temperature":0.3   
}
json_text=json.dumps(api_request,indent=2)
print(json_text)