import sqlite3

conn = sqlite3.connect("chat_memory.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM messages")
print(cursor.fetchall())