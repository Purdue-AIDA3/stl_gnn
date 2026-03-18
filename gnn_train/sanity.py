import gzip, json, re


p="results/train_ex3.jsonl.gz"
n=0
c_opt0=0; c_opt1=0; c_goal=0
pat=re.compile(r"^event:") 
with gzip.open(p,"rt") as f:
    for line in f:
        r=json.loads(line)
        strat=r.get("micp",{}).get("strategy",{})
        got0=False; got1=False
        for k,v in strat.items():
            if not v: 
                continue
            if "G0" in k or "pred=G0" in k or "G0" in str(k):
                got0=True
            if "G1" in k or "pred=G1" in k or "G1" in str(k):
                got1=True
        if got0 or got1:
            c_goal += 1
        if got0: c_opt0 += 1
        if got1: c_opt1 += 1
        n += 1
        if n>=200: break
print("first200 with any goal-marked 1s:", c_goal,"/",n)
print("first200 choose G0:", c_opt0,"/",n)
print("first200 choose G1:", c_opt1,"/",n)