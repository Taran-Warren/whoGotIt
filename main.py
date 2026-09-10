print("helllo world")
import sqlite3
import numpy as np
from model import encode_image,resize_image
import torch
connection = sqlite3.connect("objects.db")
cursor = connection.cursor()
cursor.execute("""create table if not exists objects(id integer primary key autoincrement,user_id text, label text,embedding blob,thumbnail_path text,gifter_name text)""")
#cursor.execute(""" insert into objects(user_id,label,embedding,thumbnail_path,gifter_name)values(?,?,?,?,?)""",(123,"lab",424553535,"imagge","tom"))
#connection.commit()


def get_all_objects():
    cursor.execute("""SELECT * FROM objects""")
    rows = cursor.fetchall()
    processed = []
    for row in rows:
        embedding = bytes_to_embedding(row[3])
        processed.append((row[0],row[1],row[2],embedding,row[4],row[5]))
    return processed


def add_object(user_id,label,embedding,thumbnail_path,gifter_name):
    embedding_bytes = embedding.cpu().numpy().tobytes()
    cursor.execute("""insert into objects(user_id,label,embedding,thumbnail_path,gifter_name)values(?,?,?,?,?)""",(user_id,label,embedding_bytes,thumbnail_path,gifter_name))
    connection.commit() 


def bytes_to_embedding(embedding_bytes):
    return np.frombuffer(embedding_bytes,dtype = np.float32)

def find_matches(new_image_path):
    new_embedding = encode_image(new_image_path)
    objects = get_all_objects()
    scores = []
    for obj in objects:
        stored_embedding = torch.from_numpy(obj[3])
        similarity = torch.nn.functional.cosine_similarity(new_embedding,stored_embedding)
        scores.append((obj, similarity.item()))
    scores.sort(key=lambda x:x[1],reverse=True)
    return scores[:5]
results = find_matches("test_2c.png")
print(results)    

embedding = encode_image("test_1c.png")
resize_image("test_1c.png","test_1c_r.png")
add_object(123,"mug",embedding,"test_1c_r.png","Tom")


objects = get_all_objects()
print (objects)