from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is running"}

from pydantic import BaseModel
class Item(BaseModel):
    name:str
    price:float

@app.post("/items")
def create_item(item: Item):
    new_item = {
    "name":item.name,
    "price":item.price
    }
    return {"message":"item created successfully","item":new_item}

@app.options("/items")
def options_item():
    return{"Allowed methods":["get","put","post","delete","patch"]}





# @app.get("/")
# def home():
#     return {"message": "first api is running"}

# @app.get("/health")
# def health():
#     return {"status": "healthy", "service": "rag fastapi"}

# @app.get("/items")
# def items():
#     return[
#         {"id":1,"name":"laptop","price":120000},
#         {"id":2,"name":"mobile","price":12756}
#     ]