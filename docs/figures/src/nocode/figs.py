import json,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,numpy as np
w=json.load(open('walls_by_repo.json'))
cats=['cant_do_x_missing_primitive','escape_hatch_custom_code','versioning_merge','security_auth','data_relations_schema','performance_scale','debugging_observability','determinism_concurrency','testing','ai_llm']
repos=sorted(w)
M=np.array([[len(w[r].get(c,[])) for c in cats] for r in repos])
fig,ax=plt.subplots(figsize=(11,9));im=ax.imshow(M,cmap='Blues')
ax.set_xticks(range(len(cats)),[c.replace('_',' ') for c in cats],rotation=40,ha='right');ax.set_yticks(range(len(repos)),[r.split('/')[1] for r in repos])
for i in range(len(repos)):
  for j in range(len(cats)):
    if M[i,j]: ax.text(j,i,M[i,j],ha='center',va='center',fontsize=8,color='white' if M[i,j]>12 else 'black')
ax.set_title('Wall categories in top-50 GitHub issues by reactions (keyword classifier; multi-label)');plt.colorbar(im);plt.tight_layout();plt.savefig('figures/fig1_walls_heatmap.png',dpi=130)
# fig2 Form mapping
comp={'S (state)':['table/collection','field/column type','widget props','variables/context','temp state','credentials (ref)','static data'],
'E (events)':['trigger (webhook/schedule/manual/row event)','UI event handler','message (msg/item)','form submit','inbound email','error event'],
'delta (step)':['node/action/operation','connection/wire/edge','branch/switch/router','loop/iterator','subflow/module','retry/timeout/skip_if','formula/rollup','expression binding'],
'D (log)':['execution history','run data / pinData','revisions/versionId','audit log','activity','undo stack'],
'lambda (projection)':['widget tree / component DSL','view (grid/kanban/calendar)','CSS/style tokens','pages/routes','API/OpenAPI/GraphQL schema','codegen export']}
fig,ax=plt.subplots(figsize=(12,6));ax.axis('off')
xs=np.linspace(0.05,0.85,5)
for x,(k,v) in zip(xs,comp.items()):
  ax.text(x+0.07,0.95,k,ha='center',fontsize=12,weight='bold',bbox=dict(fc='#dde8f5',ec='#335'))
  for i,p in enumerate(v): ax.text(x+0.07,0.85-i*0.09,p,ha='center',fontsize=8.5,bbox=dict(fc='white',ec='#999'))
ax.set_title('Union of no-code/low-code schema primitives mapped onto Form P = (S, E, delta, D, lambda)');plt.savefig('figures/fig2_form_mapping.png',dpi=130,bbox_inches='tight')
# fig3 escape hatch / expression risk
proj=['n8n','Langflow','Flowise','Budibase','Appsmith','Dify','Node-RED','Windmill','Kestra','Lowdefy','Directus','Activepieces']
expr=['JS (sandboxed AST)','Python','JS','Handlebars+JS','JS','Jinja2+selectors','JSONata+JS fn','JS transforms','Pebble','YAML operators (+_js)','Mustache','templating']
cves=[2,1,2,1,3,0,0,0,0,0,0,0]
fig,ax=plt.subplots(figsize=(10,5));ax.barh(proj,cves,color=['#c33' if c else '#9bb' for c in cves])
for i,e in enumerate(expr): ax.text(max(cves[i],0)+0.05,i,e,va='center',fontsize=8)
ax.set_xlabel('Critical code-execution advisories verified from primary sources (CVE.org, GHSA, OSV); not exhaustive');ax.set_title('Turing-complete escape hatches vs critical RCE advisories');ax.set_xlim(0,5);plt.tight_layout();plt.savefig('figures/fig3_escape_hatch_rce.png',dpi=130)
