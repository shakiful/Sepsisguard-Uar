"""Versioned agency-authored references and local lexical retrieval."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse
from urllib.request import Request,urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOURCES=[{'source_id':'cdc','url':'https://www.cdc.gov/sepsis/about/index.html','title':'CDC: About Sepsis'},
         {'source_id':'nigms','url':'https://www.nigms.nih.gov/education/fact-sheets/Pages/sepsis.aspx','title':'NIH NIGMS: Sepsis'}]
ALLOWED_HOSTS={'www.cdc.gov','www.nigms.nih.gov'}
BUNDLED_CORPUS=Path(__file__).with_name('reference_snapshots')


def normalize(text):
    return ' '.join(str(text).split())


def extract_agency_text(html):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,'html.parser')
    for element in soup.select('script,style,nav,header,footer,form,noscript'):
        element.decompose()
    main=soup.find('main') or soup.find('article') or soup.body or soup
    parts=[]
    for p in main.find_all(['p','li']):
        text=normalize(p.get_text(' ',strip=True))
        if len(text)>=50 and text not in parts:
            parts.append(text)
    return '\n'.join(parts)


def read_snapshot(directory):
    directory=Path(directory)
    saved=json.loads((directory/'corpus.json').read_text())
    texts={}
    for source in saved['sources']:
        relative=Path(source['text_file']);url=urlparse(source['url'])
        if relative.name!=str(relative) or relative.suffix!='.txt' or url.scheme!='https' or url.hostname not in ALLOWED_HOSTS:
            raise ValueError('Unsafe reference snapshot')
        text=(directory/relative).read_text()
        if hashlib.sha256(text.encode()).hexdigest()!=source['sha256']:
            raise ValueError('Reference snapshot changed')
        texts[source['source_id']]=(normalize(text),source)
    for passage in saved['passages']:
        if passage['source_id'] not in texts:raise ValueError('Passage has no source snapshot')
        text,source=texts[passage['source_id']]
        if passage['source_sha256']!=source['sha256'] or passage['url']!=source['url'] or normalize(passage['text']) not in text:
            raise ValueError('Retrieved passage differs from the source snapshot')
    if not texts or not saved['passages']:raise ValueError('Reference snapshot is empty')
    return saved


def build_corpus(output, *, refresh=False):
    """Reuse verified private snapshots or install the bundled corpus without HTTP.

    Live website refresh is explicit because it changes the research corpus and
    may be unavailable from Colab. Never silently substitute synthetic evidence.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    marker=output/'corpus.json'
    if not refresh:
        if marker.exists():return read_snapshot(output)
        saved=read_snapshot(BUNDLED_CORPUS)
        # Publish the manifest last, after verifying and copying all source text.
        for name in [s['text_file'] for s in saved['sources']]+['corpus.json']:
            partial=output/(name+'.partial')
            shutil.copyfile(BUNDLED_CORPUS/name,partial)
            partial.replace(output/name)
        return read_snapshot(output)
    passages=[];sources=[]
    for source in SOURCES:
        with urlopen(Request(source['url'],headers={'User-Agent':'SepsisGuard-UAR academic research'}),timeout=45) as response:
            if urlparse(response.url).hostname not in ALLOWED_HOSTS:
                raise ValueError('Reference redirected outside the agency allowlist')
            html=response.read(4*1024*1024).decode('utf-8',errors='replace')
        text=extract_agency_text(html)
        if len(text)<200 or 'sepsis' not in text.lower():
            raise ValueError('Reference page could not be extracted; do not substitute unverified evidence')
        path=output/(source['source_id']+'.txt');path.write_text(text)
        sha=hashlib.sha256(text.encode()).hexdigest()
        sources.append({**source,'text_file':path.name,'sha256':sha,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                        'scope':'Agency-authored public educational reference; not patient-specific guidance'})
        words=text.split()
        for i,start in enumerate(range(0,len(words),120)):
            chunk=' '.join(words[start:start+160])
            if len(chunk.split())<12:continue
            passages.append({'passage_id':f"{source['source_id']}:{i}:{sha[:8]}",**source,'text':chunk,'source_sha256':sha})
    result={'sources':sources,'passages':passages,'retriever':'TF-IDF cosine; local only'}
    temp=marker.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(marker)
    return result


class LocalRetriever:
    def __init__(self,passages):
        if not passages or len({p['passage_id'] for p in passages})!=len(passages):
            raise ValueError('Unique evidence passages required')
        self.passages=passages
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
        self.matrix=self.vectorizer.fit_transform([p['text'] for p in passages])

    def search(self,query,k=3):
        if not isinstance(query,str) or not 2<=len(query)<=300:
            raise ValueError('Invalid local retrieval query')
        scores=cosine_similarity(self.vectorizer.transform([query]),self.matrix).ravel()
        indices=scores.argsort()[::-1][:k]
        return [{**self.passages[int(i)],'retrieval_score':float(scores[i])} for i in indices if scores[i]>0.03]
