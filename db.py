import sqlite3 as lite
import os

auth_realm = 'ApiUser@AIMKAIST'

if not os.path.exists('db'):
    os.makedirs('db')

if not os.path.exists('db/auth.db'):
    conn = lite.connect("db/auth.db")
    conn.isolation_level = None
    cur = conn.cursor()

    create_query = \
        """
        CREATE TABLE users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username varchar(100) NOT NULL,
            password varchar(100) NOT NULL,
            description text DEFAULT ''
        )
        """
    cur.execute(create_query)
else:
    conn = lite.connect("db/auth.db")
    cur = conn.cursor()

def get_users():
    cur.execute('select username, password from users')
    udict = {}
    for username, password in cur.fetchall():
        udict[str(username)] = str(password)
    return udict