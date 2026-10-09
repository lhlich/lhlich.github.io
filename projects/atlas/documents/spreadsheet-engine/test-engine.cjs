'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Spreadsheet}=require('./engine.js');
for (const mode of ['lazy','eager']) {
  test(`${mode}: precedence, unary signs, decimals, exponents and functions`,()=>{
    const s=new Spreadsheet(mode);
    for (const [name,raw] of Object.entries({A1:'2',A2:'3',B1:'=A1+A2*4',B2:'=-(A1+A2)/2',C1:'=SUM(A1:A2, MAX(4, 1), -1)',C2:'=MIN(A1:A2, -2)+1e2'})) s.set(name,raw);
    assert.equal(s.get('B1'),14); assert.equal(s.get('B2'),-2.5);
    assert.equal(s.get('C1'),8); assert.equal(s.get('C2'),98);
    assert.equal(s.get('z99'),0);
  });
  test(`${mode}: shared dependency, transitive invalidation and no stale edge`,()=>{
    const s=new Spreadsheet(mode);
    for (const [name,raw] of Object.entries({A1:'10',B1:'=A1*2',C1:'=A1+B1',D1:'=B1+C1',E1:'7'})) s.set(name,raw);
    assert.equal(s.get('D1'),50); s.get('E1'); s.trace=[];
    s.set('A1','3'); assert.deepEqual(s.lastEdit.affected,['A1','B1','C1','D1']);
    assert.equal(s.get('D1'),15); assert.equal(s.get('E1'),7);
    assert.ok(!s.trace.includes('E1'));
    s.set('B1','9'); s.set('C1','5'); s.set('A1','100');
    assert.deepEqual(s.lastEdit.affected,['A1']); assert.equal(s.get('D1'),14);
  });
  test(`${mode}: rejected cycle and syntax writes are atomic`,()=>{
    const s=new Spreadsheet(mode); s.set('A1','2'); s.set('B1','=A1+1');
    assert.throws(()=>s.set('A1','=B1'),e=>e.code==='#CYCLE!');
    assert.equal(s.raw('A1'),'2'); assert.equal(s.get('B1'),3);
    assert.throws(()=>s.set('A1','=SUM('),e=>e.code==='#PARSE!');
    assert.deepEqual(s.dependencies('B1'),['A1']); assert.deepEqual(s.dependents('A1'),['B1']);
    assert.equal(s.get('B1'),3);
    assert.throws(()=>s.set('C1','=C1'),e=>e.code==='#CYCLE!');
    s.set('C1','=SUM(A2:B2)');
    assert.throws(()=>s.set('A2','=C1'),e=>e.code==='#CYCLE!');
  });
  test(`${mode}: errors propagate, cache and recover after source changes`,()=>{
    const s=new Spreadsheet(mode); s.set('A1','0'); s.set('B1','=4/A1'); s.set('C1','=B1+1');
    assert.equal(s.inspect('C1').code,'#DIV/0!'); assert.equal(s.inspect('C1').code,'#DIV/0!');
    s.set('A1','2'); assert.equal(s.get('C1'),3);
    s.set('D1','=1e308*1e308'); assert.equal(s.inspect('D1').code,'#NUM!');
    s.set('D1',''); assert.equal(s.get('D1'),0);
  });
  test(`${mode}: range counting, normalized names and malformed input`,()=>{
    const s=new Spreadsheet(mode); s.set('a1','2'); s.set('B1','=sum(a1:A2,A1)');
    assert.equal(s.get('b1'),4); assert.deepEqual(s.dependencies('B1'),['A1','A2']);
    for (const raw of ['=SUM(B2:A1)','=SUM(A1:ZZ999)','=2**3','=A0','=1 2','=BOGUS(1)','0x10','=1;alert(1)','=SUM()']) assert.throws(()=>s.set('C1',raw));
    assert.throws(()=>s.set('C1','=1+'.repeat(201)+'1'));
  });
}
test('lazy defers evaluation; eager resolves after the edit',()=>{
  const lazy=new Spreadsheet(); lazy.set('A1','2'); lazy.set('B1','=A1*3');
  assert.deepEqual(lazy.lastEdit.computed,[]); assert.equal(lazy.get('B1'),6);
  lazy.setMode('eager'); lazy.set('A1','4');
  assert.ok(lazy.lastEdit.computed.includes('B1')); assert.equal(lazy.get('B1'),12);
});
test('diamond DAG evaluates shared nodes once on a cold read',()=>{
  const s=new Spreadsheet(); s.set('A1','2'); s.set('B1','=A1+1'); s.set('C1','=A1+2'); s.set('D1','=B1+C1');
  s.cache.clear(); s.trace=[]; assert.equal(s.get('D1'),7);
  assert.deepEqual(s.trace,['A1','B1','C1','D1']);
});
test('random acyclic sheets agree with independent arithmetic before and after edits',()=>{
  let seed=42; const random=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/2**32;};
  for(let trial=0;trial<30;trial++) {
    const recipes=[], s=new Spreadsheet(trial%2 ? 'lazy' : 'eager');
    for(let i=0;i<25;i++) {
      if(i<3) { const value=Math.floor(random()*20); recipes.push({value}); s.set('A'+(i+1),value); }
      else { const a=Math.floor(random()*i),b=Math.floor(random()*i); recipes.push({a,b}); s.set('A'+(i+1),`=A${a+1}+A${b+1}*2`); }
    }
    const oracle=()=>{const v=[];for(const r of recipes)v.push('value'in r?r.value:v[r.a]+v[r.b]*2);return v;};
    oracle().forEach((v,i)=>assert.equal(s.get('A'+(i+1)),v));
    recipes[1]={value:7};s.set('A2','7');oracle().forEach((v,i)=>assert.equal(s.get('A'+(i+1)),v));
  }
});
