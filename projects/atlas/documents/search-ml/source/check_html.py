from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from collections import Counter
import re

ROOT=Path(__file__).resolve().parent.parent
class Page(HTMLParser):
    def __init__(self,text):
        super().__init__(); self.ids=[];self.hrefs=[];self.srcs=[];self.alt=[];self.parts=[]
        self.feed(text)
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.append(d['id'])
        if 'href' in d:self.hrefs.append(d['href'])
        if 'src' in d:self.srcs.append(d['src'])
        if tag=='img':self.alt.append(d.get('alt',''))
    def handle_data(self,data):self.parts.append(data)

files=list(ROOT.glob('*.html'))
pages={p:Page(p.read_text(encoding='utf-8')) for p in files};count=0
for p,page in pages.items():
    assert len(page.ids)==len(set(page.ids)),(p,'duplicate ids')
    assert all(page.alt),(p,'missing alt')
    assert all(s.startswith('data:') for s in page.srcs),(p,'external asset')
    assert '@@' not in p.read_text(encoding='utf-8'),(p,'unrendered token')
    for href in page.hrefs:
        u=urlsplit(href)
        if u.scheme:continue
        target=(p.parent/unquote(u.path)).resolve() if u.path else p
        assert target.exists(),(p,href,'missing file')
        if u.fragment:
            assert u.fragment in pages[target].ids,(p,href,'missing anchor')
        count+=1
    print(p.name,'IDs:',len(page.ids),'equations:',len(page.alt))
print('PASS:',count,'local links; no duplicate IDs, missing anchors, missing alt text, external assets, or template tokens')
