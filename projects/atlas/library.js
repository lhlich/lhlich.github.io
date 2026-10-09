const list = document.getElementById('documents');
const status = document.getElementById('status');
const search = document.getElementById('search');
let documents = [];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function render() {
  const query = search.value.trim().toLowerCase();
  const matches = documents.filter(doc => [doc.title, doc.description, ...(doc.topics || []), ...(doc.contents || [])].join(' ').toLowerCase().includes(query));
  list.replaceChildren();
  status.textContent = query ? (matches.length ? `${matches.length} ${matches.length === 1 ? 'document' : 'documents'} found.` : 'No notes match that search. Try another word.') : (documents.length ? '' : 'The collection is empty for now.');
  for (const doc of matches) {
    const article = element('article', 'entry');
    const aside = element('div', 'entry-aside');
    aside.append(element('p', 'entry-kind', doc.kind || (doc.format === 'pdf' ? 'PDF document' : 'Document')));
    const date = element('time', '', new Intl.DateTimeFormat('en-US', {month:'short', day:'numeric', year:'numeric', timeZone:'UTC'}).format(new Date(doc.updated + 'T00:00:00Z')));
    date.dateTime = doc.updated;
    const updated = element('div');
    updated.append(element('span', 'updated-label', 'Updated '), date);
    aside.append(updated);
    const body = element('div', 'entry-body');
    const title = element('h3');
    const url = 'reader.html?id=' + encodeURIComponent(doc.id);
    const link = element('a', '', doc.title);
    link.href = url;
    title.append(link);
    body.append(title, element('p', 'description', doc.description));
    if (doc.details) body.append(element('p', 'details', doc.details));
    if (doc.contents?.length) {
      const contents = element('div', 'contents');
      const subjects = element('ul');
      for (const subject of doc.contents) subjects.append(element('li', '', subject));
      contents.append(element('h4', '', 'Inside'), subjects);
      body.append(contents);
    }
    const read = element('a', 'read', 'Read ' + (doc.kind === 'Study guide' ? 'the guide' : 'document') + ' →');
    read.href = url;
    read.setAttribute('aria-label', 'Read ' + doc.title);
    body.append(read);
    article.append(aside, body);
    list.append(article);
  }
}

search.disabled = true;
search.addEventListener('input', render);
fetch('catalog.json').then(response => {
  if (!response.ok) throw new Error('Catalog unavailable');
  return response.json();
}).then(catalog => {
  documents = catalog.documents.sort((a,b) => b.updated.localeCompare(a.updated));
  render();
  search.disabled = false;
}).catch(() => { status.textContent = 'The collection could not be loaded. Please refresh, or use the catalog link below.'; });
