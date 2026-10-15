import re,json,sys
def repair(b):
    fixes=[]
    t=re.sub(r'\s*//\s*<\d+>','',b)
    t2=re.sub(r'\{\s*\.\.\.\s*\}','{}',t)
    if t2!=t: fixes.append('elided object {...} -> {}')
    t=t2
    t2=re.sub(r'\n\s*\.\.\.\s*\n','\n',t)
    if t2!=t: fixes.append('elided members ... removed')
    t=t2
    t2=re.sub(r',(\s*[}\]])',r'\1',t)
    if t2!=t: fixes.append('trailing comma removed')
    t=t2
    t2=t.replace('{debezium-version}','3.0.0.Final')
    if t2!=t: fixes.append('{debezium-version} -> 3.0.0.Final')
    t=t2
    d=t.count('{')-t.count('}')
    if d>0: t=t+'}'*d; fixes.append('missing closing brace added')
    return t,fixes
out=[]
for p in sys.argv[1:]:
    s=open(p).read()
    for m in re.finditer(r'\[source,json[^\]]*\]\n----\n(.*?)\n----', s, re.S):
        b=m.group(1)
        if '"op"' not in b: continue
        t,f=repair(b)
        try: j=json.loads(t)
        except Exception as e: print('FAIL',p,m.start(),e); continue
        line=s[:m.start()].count('\n')+1
        out.append({"file":p,"line":line,"fixes":f,"event":j})
for o in out: 
    pl=o['event'].get('payload',o['event'])
    print(o['file'],o['line'],pl.get('op'),o['fixes'])
json.dump(out,open('debezium_examples.json','w'),indent=1)
