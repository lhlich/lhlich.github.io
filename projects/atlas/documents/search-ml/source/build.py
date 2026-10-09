from pathlib import Path
import ast, base64, html, io, re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).parent
OUT=ROOT.parent
OUT.mkdir(exist_ok=True)
PAGES=[('index','Start here','Search & ML Study Notes','Search · modeling · systems · implementation'),
       ('search','01 · Search','Search technologies','Indexes, candidate generation, and reranking'),
       ('modeling','02 · Modeling','Modeling and evaluation','Targets, losses, metrics, and observation bias'),
       ('system','03 · System','Two on-device search designs','Data, training, serving, and measurement'),
       ('efficiency','04 · Efficiency','Device efficiency','Representation, adapters, caches, and measurement'),
       ('coding','05 · Coding','Coding and ML fundamentals','Invariants, stable numerics, and training diagnosis'),
       ('glossary','Reference','Glossary and sources','Notation, distinctions, and primary reading')]
EQ={
'bm25':[(r'\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{IDF}(t)\,\frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1(1-b+b|d|/\mathrm{avgdl})}', 'BM25 is the sum over query terms of IDF times f*(k1+1)/(f+k1*(1-b+b*document_length/average_length)).'),(r'\mathrm{IDF}(t)=\log\left(1+\frac{N-\mathrm{df}_t+0.5}{\mathrm{df}_t+0.5}\right)', 'Positive IDF variant: log(1+(N-df+0.5)/(df+0.5)).')],
'bce':[(r'\ell=-y\log p-(1-y)\log(1-p)', 'Binary cross-entropy equals -y log p -(1-y) log(1-p).'),(r'\frac{\partial\ell}{\partial z}=p-y,\qquad \nabla_w\ell=(p-y)x', 'Logit gradient is p-y; weight gradient is (p-y)x.')],
'ce':[(r'\ell=-\sum_j y_j\log p_j,\qquad\frac{\partial\ell}{\partial z_j}=p_j-y_j', 'Multiclass cross-entropy: negative sum y_j log p_j; derivative with respect to z_j is p_j-y_j.')],
'contrastive':[(r'\ell=-\log\frac{\exp(s(q,d_+)/\tau)}{\sum_{j=1}^{K}\exp(s(q,d_j)/\tau)}', 'Contrastive cross-entropy: negative log of positive exp(score/temperature) divided by sum of candidate exp(score/temperature).')],
'ndcg':[(r'\mathrm{DCG}@k=\sum_{i=1}^{k}\frac{2^{r_i}-1}{\log_2(i+1)},\qquad\mathrm{NDCG}@k=\frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}', 'DCG sums (2^relevance-1)/log2(rank+1); NDCG divides by ideal DCG at the same cutoff.')],
'ips':[(r'\widehat V(\pi)=\frac{1}{n}\sum_{i=1}^{n}\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}r_i', 'Off-policy value estimate is the mean of reward times target action probability divided by logging action probability.')],
'prefix':[(r'\mathrm{sum}=P[r_2+1,c_2+1]-P[r_1,c_2+1]', 'Rectangle sum starts with P[r2+1,c2+1] minus P[r1,c2+1].'),(r'\qquad -P[r_2+1,c_1]+P[r_1,c_1]', 'Then subtract P[r2+1,c1] and add P[r1,c1].')],
'attention':[(r'\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V', 'Attention is softmax over keys of QK transpose divided by square root of key width, plus mask M, then multiplied by V.')]
}
# Split long equations so a phone can show the complete expression.
EQ['bm25']=[
 (r'\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{IDF}(t)F(t,d)', 'BM25 sums IDF(t) times the frequency-length factor F(t,d).'),
 (r'F(t,d)=\frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1(1-b+b|d|/\mathrm{avgdl})}', 'F(t,d)=f*(k1+1)/(f+k1*(1-b+b*document_length/average_length)).'),
 EQ['bm25'][1]]
EQ['ndcg']=[
 (r'\mathrm{DCG}@k=\sum_{i=1}^{k}\frac{2^{r_i}-1}{\log_2(i+1)}', 'DCG sums (2^relevance-1)/log2(rank+1).'),
 (r'\mathrm{NDCG}@k=\frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}', 'NDCG is DCG divided by ideal DCG at the same cutoff.')]
EQ['ce']=[
 (r'\ell=-\sum_j y_j\log p_j', 'Cross-entropy is negative sum y_j log p_j.'),
 (r'\frac{\partial\ell}{\partial z_j}=p_j-y_j', 'Logit gradient is probability minus target.')]
CSS='''
:root{--ink:#1e293b;--muted:#566679;--accent:#11696a;--paper:#fff;--wash:#f3f6f5;--line:#d5dfdf;--serif:Georgia,"Times New Roman",serif}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:95px}body{margin:0;color:var(--ink);background:var(--wash);font:17px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:#08656b;text-underline-offset:3px}a:hover{color:#073e43}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #cb7800;outline-offset:4px}.skip{position:absolute;top:-100px;left:20px;background:white;padding:8px;z-index:9}.skip:focus{top:10px}.top{background:#143b3c;color:#fff;padding:18px 5vw}.brand{max-width:1160px;margin:auto;display:flex;justify-content:space-between;gap:20px;align-items:center}.home-link{color:#fff}.home-link:hover{color:#d4e5e3}.brand strong{letter-spacing:.07em;font-size:13px;text-transform:uppercase}.brand span{font-size:12px;color:#d4e5e3}.nav{background:white;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}.nav-inner{max-width:1160px;margin:auto;display:flex;flex-wrap:wrap;gap:5px;padding:9px 15px}.nav a{padding:6px 11px;border-radius:6px;font-size:13px;text-decoration:none;font-weight:650}.nav a[aria-current=page]{background:#e3efec;color:#08474b}.layout{max-width:1160px;margin:34px auto;display:grid;grid-template-columns:210px minmax(0,1fr);gap:35px;padding:0 20px}.toc{font-size:13px;position:sticky;top:90px;align-self:start;max-height:calc(100vh - 120px);overflow:auto}.toc strong{display:block;text-transform:uppercase;letter-spacing:.07em;font-size:11px;margin-bottom:12px;color:var(--muted)}.toc a{display:block;text-decoration:none;padding:5px 0;line-height:1.45}.toc .utility{border-top:1px solid var(--line);margin-top:15px;padding-top:12px}main{min-width:0;background:white;border:1px solid var(--line);border-radius:12px;padding:42px 46px 35px;box-shadow:0 6px 25px #193a3b05}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:11px;font-weight:750;color:var(--accent)}h1{font-family:var(--serif);font-size:42px;line-height:1.12;letter-spacing:-.03em;margin:12px 0 16px}h2{font-family:var(--serif);font-size:27px;line-height:1.25;margin:42px 0 15px;scroll-margin-top:100px}h3{font-size:17px;line-height:1.4;margin:0 0 10px}p{margin:0 0 17px}.subtitle{font-size:15px;color:var(--muted);margin-bottom:26px}.lead{font:21px/1.55 var(--serif);color:#334b55}.goal,.example,.note,.path{padding:18px 21px;border-radius:7px;margin:24px 0}.goal{border-left:4px solid var(--accent);background:#edf6f2}.example{background:#f4f6fa;border:1px solid #dae0e9}.note{background:#fff8e9;border-left:4px solid #c89432}.path{border:1px solid var(--line);background:#f7faf8}.goal p:last-child,.example p:last-child,.note p:last-child,.path p:last-child{margin-bottom:0}.table-wrap{overflow-x:auto;margin:23px 0;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}th{text-align:left;background:#edf2f1;color:#294747;font-size:12px;text-transform:uppercase;letter-spacing:.03em}td,th{padding:12px 13px;vertical-align:top;border-bottom:1px solid var(--line)}tr:last-child td{border-bottom:0}td:first-child{font-weight:550}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.86em;background:#edf1f3;border-radius:3px;padding:2px 4px;overflow-wrap:anywhere}pre{background:#132b35;color:#eef7f4;padding:20px;border-radius:7px;font-size:13px;line-height:1.6;overflow-x:auto;tab-size:4}pre code{font-size:inherit;background:none;padding:0;color:inherit;overflow-wrap:normal}.equations{overflow-x:auto;padding:17px 4px;margin:22px 0;background:#fafcfc;border-top:1px solid #e4ecea;border-bottom:1px solid #e4ecea}.equations img{display:block;max-width:100%;margin:8px auto;height:auto}details{border:1px solid #c8d9d5;border-radius:7px;padding:16px 20px;margin:25px 0;background:#f7fbf9}summary{font-weight:650;cursor:pointer;color:#12595c}details[open] summary{margin-bottom:15px}li{margin-bottom:9px}.cite{white-space:normal;font-size:12px}button{font:inherit;font-size:13px;background:#e4eeeb;color:#164747;border:1px solid #c3d5cf;border-radius:5px;padding:8px 12px;cursor:pointer}.actions{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}.next{margin-top:32px;padding-top:18px;border-top:1px solid var(--line);font-size:14px}.footer{font-size:12px;color:var(--muted);padding-top:25px}.chapter+.chapter{margin-top:70px;padding-top:35px;border-top:3px solid #aac9bf}.scroll-hint{display:none}.section-marker{color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.complete-code{margin-top:40px}
@media(max-width:900px){.layout{grid-template-columns:1fr;gap:18px;margin:20px auto}.toc{position:static;max-height:none;border-bottom:1px solid var(--line);padding-bottom:18px}.toc a{display:inline-block;margin-right:14px}.toc .utility{display:none}main{padding:30px}.brand span{display:none}.nav-inner{gap:1px}.nav a{padding:5px 8px;font-size:12px}}
@media(max-width:520px){.scroll-hint{display:block;font-size:11px;color:var(--muted);margin:10px 0 3px}body{font-size:16px;line-height:1.65}.layout{padding:0 10px}.top{padding:15px}.nav{position:static}main{padding:25px 18px;border-radius:8px}h1{font-size:34px}h2{font-size:25px;margin-top:35px}.lead{font-size:20px}.goal,.example,.note,.path{padding:15px}.toc{font-size:12px}.table-wrap table{min-width:560px}pre{font-size:12px;padding:16px}.equations img{max-width:100%}.nav-inner{padding:7px}html{scroll-padding-top:20px}}
@media print{@page{size:A4;margin:18mm 15mm}body{background:white;font-size:10pt;line-height:1.5;color:#172a33}.top,.nav,.toc,.actions,.skip,.footer{display:none}.layout{display:block;max-width:none;margin:0;padding:0}main{padding:0;border:0;box-shadow:none;border-radius:0}h1{font-size:29pt}h2{font-size:19pt;margin-top:22pt;break-after:avoid}h3{break-after:avoid}.lead{font-size:13pt}.chapter+.chapter{break-before:page;margin-top:0;padding-top:0;border:0}.goal,.example,.note,.path{padding:10pt;margin:14pt 0}.table-wrap{overflow:visible}table{font-size:9pt;min-width:0!important}tr{break-inside:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f5;color:#142834;font-size:8pt;border:1px solid #cbd3d4}pre code{color:inherit}.equations{overflow:visible}.equations img{max-width:100%!important}details{break-inside:auto}details>*{display:block}summary{display:block}.next{display:none}a{color:inherit}.subtitle{font-size:10pt}.complete-code{break-before:page}}
'''
JS='''
function setAnswers(open){document.querySelectorAll('details').forEach(d=>d.open=open)}
let prior=[];
window.addEventListener('beforeprint',()=>{prior=[...document.querySelectorAll('details')].map(d=>d.open);setAnswers(true)});
window.addEventListener('afterprint',()=>{document.querySelectorAll('details').forEach((d,i)=>d.open=prior[i])});
'''
source=(ROOT/'exercises.py').read_text(encoding='utf-8')
lines=source.splitlines()
code={n.name:'\n'.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

drill_source=(ROOT/'search_drills.py').read_text(encoding='utf-8')
drill_lines=drill_source.splitlines()
code.update({n.name:'\n'.join(drill_lines[n.lineno-1:n.end_lineno]) for n in ast.parse(drill_source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})

def equation(name):
    tags=[]
    for latex,alt in EQ[name]:
        fig=plt.figure(figsize=(10,.9))
        text=fig.text(.02,.35,'$'+latex+'$',fontsize=16,color='#203943')
        fig.canvas.draw()
        extent=text.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        buf=io.BytesIO()
        fig.savefig(buf,format='svg',bbox_inches=extent.expanded(1.02,1.25),transparent=True)
        plt.close(fig)
        raw=buf.getvalue()
        w=float(re.search(rb'width="([\d.]+)pt"',raw).group(1))*1.17
        tags.append('<img style="width:'+str(round(w))+'px" alt="'+html.escape(alt,quote=True)+'" src="data:image/svg+xml;base64,'+base64.b64encode(raw).decode()+'">')
    return '<div class="equations" role="group" aria-label="Equation">'+''.join(tags)+'</div>'

def build_content(slug):
    body=(ROOT/'chapters'/f'{slug}.html').read_text(encoding='utf-8')
    body=re.sub(r'@@EQ:(\w+)@@',lambda m:equation(m[1]),body)
    body=re.sub(r'@@CODE:(\w+)@@',lambda m:'<p class="scroll-hint">Code · swipe horizontally to read long lines</p><pre tabindex="0"><code>'+html.escape(code[m[1]])+'</code></pre>',body)
    body=body.replace('<table>','<p class="scroll-hint">Table · swipe horizontally for all columns</p><div class="table-wrap" role="region" aria-label="Comparison table" tabindex="0"><table>').replace('</table>','</table></div>')
    return body

def nav(current,single=False):
    return '<nav class="nav" aria-label="Chapters"><div class="nav-inner">'+''.join('<a '+('aria-current="page" ' if current==s else '')+'href="'+('#'+s if single else s+'.html')+'">'+n+'</a>' for s,n,_,_ in PAGES)+'</div></nav>'

def template(title,current,body,toc,single=False):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Technical study guide: search, modeling, evaluation, systems, and coding."><title>'+html.escape(title)+' — Search &amp; ML Study Notes</title><style>'+CSS+'</style></head><body><a class="skip" href="#main">Skip to content</a><header class="top"><div class="brand"><strong>Search &amp; ML · Study notes</strong><span>Mechanisms · examples · exercises</span></div></header>'+nav(current,single)+'<div class="layout"><aside class="toc" aria-label="On this page"><strong>On this page</strong>'+toc+'<div class="utility">Offline-ready · primary references<br>Read → reconstruct → test</div></aside><main id="main">'+body+'<footer class="footer">Search &amp; ML Study Notes · October 2026 · Worked examples are illustrative.</footer></main></div><script>'+JS+'</script></body></html>'

contents={s:build_content(s) for s,_,_,_ in PAGES}
for s,n,title,sub in PAGES:
    body='<div class="eyebrow">'+n+'</div><h1>'+title+'</h1><p class="subtitle">'+sub+'</p><div class="actions"><button onclick="setAnswers(true)">Show solutions</button><button onclick="setAnswers(false)">Hide solutions</button><button onclick="window.print()">Print chapter</button></div>'+contents[s]
    toc=''.join('<a href="#'+id+'">'+re.sub('<[^>]+>','',name)+'</a>' for id,name in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',contents[s]))
    if '<details>' not in contents[s]:
        body=body.replace('<button onclick="setAnswers(true)">Show solutions</button><button onclick="setAnswers(false)">Hide solutions</button>','')
    (OUT/f'{s}.html').write_text(template(title,s,body,toc), encoding='utf-8')

combined=[]
for s,n,title,sub in PAGES:
    body=contents[s]
    body=re.sub(r'id="([^"]+)"',lambda m:'id="'+s+'-'+m[1]+'"',body)
    def fix(m):
        target,frag=m[1],m[2]
        return 'href="#'+target+('-'+frag if frag else '')+'"'
    body=re.sub(r'href="(index|search|modeling|system|efficiency|coding|glossary)\.html(?:#([^"]+))?"',fix,body)
    combined.append('<section class="chapter" id="'+s+'"><div class="eyebrow">'+n+'</div><h1>'+title+'</h1><p class="subtitle">'+sub+'</p>'+body+'</section>')
combined.append('<section class="complete-code" id="complete-code"><h2>Complete runnable code</h2><p>Copy each block into the named file. Requires Python 3 and NumPy. These are the same files included in the linked-chapter package.</p><details><summary>exercises.py</summary><pre><code>'+html.escape(source)+'</code></pre></details><details><summary>test_exercises.py</summary><pre><code>'+html.escape((ROOT/'test_exercises.py').read_text(encoding='utf-8'))+'</code></pre></details></section>')
for filename in ['search_drills.py','search_drills_blank.py','test_search_drills.py','test_regressions.py']:
    combined.append('<section class="complete-code"><h2>'+filename+'</h2><details><summary>Show file</summary><pre><code>'+html.escape((ROOT/filename).read_text(encoding='utf-8'))+'</code></pre></details></section>')
body='<div class="actions"><button onclick="setAnswers(true)">Show solutions</button><button onclick="setAnswers(false)">Hide solutions</button><button onclick="window.print()">Print full guide</button></div>'+''.join(combined)
toc=''.join('<a href="#'+s+'">'+n+'</a>' for s,n,_,_ in PAGES)+'<a href="#complete-code">Complete runnable code</a>'
(OUT/'Study-Guide.html').write_text(template('Complete reader','',body,toc,True), encoding='utf-8')
for name in ['exercises.py','test_exercises.py','search_drills.py','search_drills_blank.py','test_search_drills.py','test_regressions.py']:
    (OUT/name).write_text((ROOT/name).read_text(encoding='utf-8'), encoding='utf-8')

print('Built seven linked pages and one self-contained reader.')
