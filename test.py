import datetime

filename = datetime.datetime.now().isoformat()

f = open(filename, "w")
f.write(filename)
f.close()
