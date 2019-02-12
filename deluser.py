#!/usr/bin/python
import sys, getopt
import db, hashlib
import getpass, sqlite3

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"d:")
    except getopt.GetoptError:
      print 'deluser.py -d username'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-d':
         username = arg
         break
    else:
        username = raw_input("username:").strip()

    if len(username) == 0:
        print("invalid username")
        return 1

    fail = False

    try:
        c = db.conn.cursor()
        c.execute('select username from users where username=\'%s\''% username)
        res = c.fetchall()
        db.conn.commit()
    except sqlite3.Error:
        c.execute("rollback")

    if len(res) == 0:
        print ("ID does not exist")
        return 1
    else:
        if (raw_input("really want to delete?[y/n]:") == 'y'):
            try:
                c = db.conn.cursor()
                c.execute('delete from users where username=\'%s\'' % username)
                db.conn.commit()
            except sqlite3.Error:
                c.execute("rollback")
            print ("delete complete")
        else:
            print ("canceled")

    print ("restart the server program for update")
    return 0

if __name__ == "__main__":
   main(sys.argv[1:])