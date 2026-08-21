print("helllo world")
import sqlite3
connection = sqlite3.connect("objects.db")
cursor = connection.cursor()
cursor.execute("""create table if not exists objects(id integer primary key autoincrement,user_id text, label text,embedding blob,thumbnail_path text,gifter_name text)""")
cursor.execute(""" insert into objects(user_id,label,embedding,thumbnail_path,gifter_name)values(?,?,?,?,?)""",(123,"lab",424553535,"imagge","tom"))
connection.commit()