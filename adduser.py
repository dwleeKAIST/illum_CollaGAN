#!/usr/bin/python
import sys, getopt
import db, hashlib
import getpass, sqlite3

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"a:")
    except getopt.GetoptError:
        print('adduser.py -a <username>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-a':
            username = arg
            break
    else:
        username = raw_input("username:").strip()
    password = getpass.getpass("password:")
    #description = raw_input("Description: ").strip()
    description = "BISPL-KAIST user2"#raw_input("Description: ").strip()

    if len(username) == 0:
        print("invalid username")
        return 1
    if len(password) == 0:
        print("invalid password")
        return 1

    conc = ':'.join([username,db.auth_realm,password])

    m = hashlib.md5()
    m.update(conc)
    password = m.hexdigest()

    fail = False

    try:
        c = db.conn.cursor()
        c.execute('select username from users where username=\'%s\''% username)

        if len(c.fetchall()) == 0:
            c.execute("insert into users(username, password, description) values ('%s', '%s', '%s')"
                           % (username, password, description))
        else:
            fail = True
        db.conn.commit()
    except sqlite3.Error:
        c.execute("rollback")

    if fail:
        print ("ID already exists")
        return 1
    print ("restart the server program for update")
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])