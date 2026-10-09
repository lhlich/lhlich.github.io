(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const names = Array.from({length:16}, (_,i) => 'ABCD'[i%4]+(1+Math.floor(i/4)));
  let sheet, selected='A1', affected=new Set();
  const buttons=new Map();
  const head = text => { const node=document.createElement('span'); node.className='head'; node.textContent=text; return node; };
  $('grid').append(head(''),...Array.from('ABCD',head));
  for (let row=1;row<=4;row++) {
    $('grid').append(head(row));
    for (const column of 'ABCD') {
      const name=column+row, button=document.createElement('button');
      button.type='button'; button.className='cell';
      const label=document.createElement('span'), value=document.createElement('span');
      label.className='addr'; label.textContent=name; value.className='value';
      button.append(label,value); button.addEventListener('click',()=>select(name));
      buttons.set(name,{button,value}); $('grid').append(button);
      const option=document.createElement('option'); option.value=name; option.textContent=name; $('cell').append(option);
    }
  }
  function select(name) { selected=name; $('cell').value=name; $('input').value=sheet.raw(name); render(); }
  function status(text,error=false) { $('status').textContent=text; $('status').className=error?'error':''; }
  function list(id,items,empty) {
    const nodes=(items.length?items:[empty]).map(text=>{ const li=document.createElement('li'); li.textContent=text; return li; });
    $(id).replaceChildren(...nodes);
  }
  function render() {
    for (const [name,{button,value}] of buttons) {
      const cached=sheet.cache.get(name);
      const display=cached ? (cached.ok?String(Number(cached.value.toPrecision(10))):cached.code) : 'needs read';
      value.textContent=display; value.className='value'+(!cached?' pending':'');
      button.classList.toggle('affected',affected.has(name)); button.classList.toggle('selected',name===selected);
      button.setAttribute('aria-pressed',String(name===selected));
      button.setAttribute('aria-label',name+': '+display+(affected.has(name)?', affected':''));
      button.title=sheet.raw(name)||'(blank = 0)';
    }
    $('dependencies').textContent=sheet.dependencies(selected).join(', ')||'None';
    $('dependents').textContent=sheet.dependents(selected).join(', ')||'None';
    $('affected').textContent=[...affected].sort().join(', ')||'None';
    const edges=[];
    for (const name of names) for (const dep of sheet.dependencies(name)) edges.push(dep+' → '+name);
    list('edges',edges,'No dependency edges.');
    list('trace',sheet.trace,'None — values came from cache, or no read was requested.');
  }
  function reset(kind) {
    sheet=new SheetEngine.Spreadsheet($('mode').value);
    const values=kind==='diamond'?{A1:'10',B1:'=A1*2',C1:'=A1+5',D1:'=B1+C1',D4:'99'}:{A1:'10',A2:'5',B1:'=SUM(A1:A2)',C1:'=B1*2',D1:'=C1+1',D4:'99'};
    for (const [name,raw] of Object.entries(values)) sheet.set(name,raw);
    for (const name of names) sheet.inspect(name);
    sheet.trace=[]; affected=new Set(); select('A1');
    status('Loaded '+kind+'. All visible cells have been read to establish a baseline.');
  }
  $('cell').addEventListener('change',()=>select($('cell').value));
  $('edit').addEventListener('submit',event=>{
    event.preventDefault();
    try {
      const edit=sheet.set(selected,$('input').value); affected=new Set(edit.affected);
      status('Saved '+selected+'. '+edit.affected.length+' affected; '+edit.computed.length+' calculated during the write.');
    } catch (error) {
      if (!(error instanceof SheetEngine.SheetError)) throw error;
      sheet.trace=[]; status(error.code+' '+error.message+' Previous input retained: '+(sheet.raw(selected)||'(blank)'),true);
    }
    render();
  });
  $('mode').addEventListener('change',()=>{sheet.setMode($('mode').value);status('Switched to '+sheet.mode+' mode. '+sheet.trace.length+' new evaluations.');render();});
  $('read-selected').addEventListener('click',()=>{
    sheet.trace=[]; const result=sheet.inspect(selected);
    status(result.ok?'Read '+selected+' = '+result.value:result.code+' '+result.message,!result.ok); render();
  });
  $('read-all').addEventListener('click',()=>{sheet.trace=[];for(const name of names)sheet.inspect(name);status('Read all 16 visible cells. '+sheet.trace.length+' new evaluations.');render();});
  $('chain').addEventListener('click',()=>reset('chain'));
  $('diamond').addEventListener('click',()=>reset('diamond'));
  reset('chain');
})();
