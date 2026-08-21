print("helllo world")
import sqlite3
connection = sqlite3.connect("objects.db")
cursor = connection.cursor()
cursor.execute("""create table if not exists objects(id integer primary key autoincrement,user_id text, label text,embedding blob,thumnail_path text, gifter_name text)""")
