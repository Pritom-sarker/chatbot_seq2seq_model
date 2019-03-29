import re
s="I can't date her sister until that one gets a boyfriend.  And that's the catch. She doesn't want a boyfriend."

replaced = re.sub("['t]", '000', s)
print(s.replace("'t","s"))
print (replaced )