/* Original teaching implementation. No dependencies. Run with Node or load in a browser. */
(function (root) {
  'use strict';
  class SheetError extends Error {
    constructor(code, message) { super(message); this.name = 'SheetError'; this.code = code; }
  }
  const fail = (code, text) => { throw new SheetError(code, text); };
  function address(input) {
    const value = String(input).trim().toUpperCase();
    if (!/^[A-Z]{1,2}[1-9]\d{0,2}$/.test(value)) fail('#REF!', 'Use a cell from A1 through ZZ999.');
    return value;
  }
  function coordinates(ref) {
    const [, letters, row] = address(ref).match(/^([A-Z]+)(\d+)$/);
    let column = 0;
    for (const letter of letters) column = column*26 + letter.charCodeAt(0)-64;
    return [column, Number(row)];
  }
  function columnName(n) {
    let result = '';
    while (n) { n--; result = String.fromCharCode(65+n%26)+result; n = Math.floor(n/26); }
    return result;
  }
  function range(start, end) {
    const [c1,r1] = coordinates(start), [c2,r2] = coordinates(end);
    if (c1>c2 || r1>r2) fail('#REF!', 'Range endpoints must run from top-left to bottom-right.');
    if ((c2-c1+1)*(r2-r1+1)>256) fail('#LIMIT!', 'A range may contain at most 256 cells.');
    const result = [];
    for (let r=r1;r<=r2;r++) for (let c=c1;c<=c2;c++) result.push(columnName(c)+r);
    return result;
  }
  function tokenize(text) {
    const tokens = []; let i=0;
    if (text.length>2000) fail('#LIMIT!', 'A formula may contain at most 2,000 characters.');
    while (i<text.length) {
      const rest = text.slice(i);
      if (/^\s/.test(rest)) { i++; continue; }
      const num = rest.match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
      const word = rest.match(/^[A-Za-z][A-Za-z0-9]*/);
      if (num) { tokens.push({type:'number',value:Number(num[0])}); i+=num[0].length; }
      else if (word) { tokens.push({type:'word',value:word[0].toUpperCase()}); i+=word[0].length; }
      else if ('+-*/(),:'.includes(text[i])) { tokens.push({type:text[i],value:text[i]}); i++; }
      else fail('#PARSE!', 'Unsupported character: '+text[i]);
      if (tokens.length>200) fail('#LIMIT!', 'A formula may contain at most 200 tokens.');
    }
    tokens.push({type:'end'}); return tokens;
  }
  function parse(text) {
    const tokens = tokenize(text); let index=0;
    const peek = () => tokens[index];
    const take = type => {
      if (peek().type!==type) fail('#PARSE!', 'Expected '+type+', found '+peek().type+'.');
      return tokens[index++];
    };
    function expression() {
      let node = product();
      while (['+','-'].includes(peek().type)) { const op=tokens[index++].type; node={type:'binary',op,left:node,right:product()}; }
      return node;
    }
    function product() {
      let node = unary();
      while (['*','/'].includes(peek().type)) { const op=tokens[index++].type; node={type:'binary',op,left:node,right:unary()}; }
      return node;
    }
    function unary() {
      if (['+','-'].includes(peek().type)) { const op=tokens[index++].type; return {type:'unary',op,child:unary()}; }
      return primary();
    }
    function primary() {
      if (peek().type==='number') return {type:'number',value:take('number').value};
      if (peek().type==='(') { take('('); const node=expression(); take(')'); return node; }
      if (peek().type==='word') {
        const name=take('word').value;
        if (peek().type==='(') {
          if (!['SUM','MIN','MAX'].includes(name)) fail('#PARSE!', 'Supported functions: SUM, MIN, MAX.');
          take('('); const args=[];
          if (peek().type===')') fail('#PARSE!', 'Functions need at least one argument.');
          while (true) {
            if (peek().type==='word' && tokens[index+1]?.type===':') {
              const first=take('word').value; take(':'); const last=take('word').value;
              args.push({type:'range',cells:range(first,last)});
            } else args.push(expression());
            if (peek().type!==',') break;
            take(',');
          }
          take(')'); return {type:'call',name,args};
        }
        return {type:'ref',name:address(name)};
      }
      fail('#PARSE!', 'Expected a number, cell, function, or parenthesized expression.');
    }
    const ast=expression(); take('end');
    const dependencies=new Set(); let expanded=0;
    function walk(node) {
      if (node.type==='ref') { dependencies.add(node.name); expanded++; }
      if (node.type==='range') for (const cell of node.cells) { dependencies.add(cell); expanded++; }
      if (node.type==='binary') { walk(node.left); walk(node.right); }
      if (node.type==='unary') walk(node.child);
      if (node.type==='call') node.args.forEach(walk);
    }
    walk(ast);
    if (expanded>512) fail('#LIMIT!', 'A formula may reference at most 512 expanded cells.');
    return {ast,dependencies};
  }
  function compile(raw) {
    const trimmed=raw.trim();
    if (trimmed.startsWith('=')) return parse(trimmed.slice(1));
    if (trimmed && !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) fail('#PARSE!', 'Enter a number, blank, or a formula beginning with =.');
    const value=trimmed ? Number(trimmed) : 0;
    if (!Number.isFinite(value)) fail('#NUM!', 'Numbers must be finite.');
    return {ast:{type:'number',value},dependencies:new Set()};
  }
  class Spreadsheet {
    constructor(mode='lazy') {
      if (!['lazy','eager'].includes(mode)) fail('#MODE!', 'Choose lazy or eager.');
      this.mode=mode; this.cells=new Map(); this.reverse=new Map(); this.cache=new Map(); this.trace=[]; this.lastEdit=null;
    }
    raw(name) { return this.cells.get(address(name))?.raw || ''; }
    dependencies(name) { return [...(this.cells.get(address(name))?.dependencies || [])].sort(); }
    dependents(name) { return [...(this.reverse.get(address(name)) || [])].sort(); }
    affected(name) {
      const found=new Set(), pending=[address(name)];
      while (pending.length) {
        const cell=pending.pop();
        if (found.has(cell)) continue;
        found.add(cell); pending.push(...(this.reverse.get(cell) || []));
      }
      return found;
    }
    set(name, input) {
      name=address(name); const raw=String(input);
      const compiled=compile(raw); // Parse and validate before mutating any state.
      if (!this.cells.has(name) && this.cells.size>=512) fail('#LIMIT!', 'At most 512 stored cells are supported.');
      const complete=new Set(), visiting=new Set(), path=[];
      const check=cell => {
        if (visiting.has(cell)) fail('#CYCLE!', 'Circular reference: '+[...path,cell].join(' → '));
        if (complete.has(cell)) return;
        if (path.length>=128) fail('#LIMIT!', 'Dependency paths may contain at most 128 cells.');
        visiting.add(cell); path.push(cell);
        const deps=cell===name ? compiled.dependencies : (this.cells.get(cell)?.dependencies || []);
        for (const dep of deps) check(dep);
        path.pop(); visiting.delete(cell); complete.add(cell);
      };
      // Check every root: an edit can lengthen an existing dependent's path.
      for (const cell of new Set([...this.cells.keys(),name])) { complete.clear(); check(cell); }
      const affected=this.affected(name);
      for (const dep of this.cells.get(name)?.dependencies || []) {
        this.reverse.get(dep)?.delete(name);
        if (!this.reverse.get(dep)?.size) this.reverse.delete(dep);
      }
      this.cells.set(name,{raw,...compiled});
      for (const dep of compiled.dependencies) {
        if (!this.reverse.has(dep)) this.reverse.set(dep,new Set());
        this.reverse.get(dep).add(name);
      }
      for (const cell of affected) this.cache.delete(cell);
      this.trace=[];
      if (this.mode==='eager') for (const cell of [...affected].sort()) this.inspect(cell);
      this.lastEdit={cell:name,affected:[...affected].sort(),computed:[...this.trace]};
      return this.lastEdit;
    }
    setMode(mode) {
      if (!['lazy','eager'].includes(mode)) fail('#MODE!', 'Choose lazy or eager.');
      this.mode=mode; this.trace=[];
      if (mode==='eager') for (const cell of [...this.cells.keys()].sort()) this.inspect(cell);
    }
    get(name) { return this._resolve(address(name),new Set()); }
    inspect(name) {
      try { return {ok:true,value:this.get(name)}; }
      catch (error) { if (!(error instanceof SheetError)) throw error; return {ok:false,code:error.code,message:error.message}; }
    }
    _resolve(name, visiting) {
      if (this.cache.has(name)) {
        const entry=this.cache.get(name);
        if (!entry.ok) fail(entry.code,entry.message);
        return entry.value;
      }
      if (visiting.has(name)) fail('#CYCLE!', 'Circular reference at '+name);
      visiting.add(name);
      try {
        const ast=this.cells.get(name)?.ast || {type:'number',value:0};
        const value=this._evaluate(ast,visiting);
        if (!Number.isFinite(value)) fail('#NUM!', 'Non-finite result in '+name);
        this.cache.set(name,{ok:true,value}); this.trace.push(name);
        return value;
      } catch (error) {
        if (!(error instanceof SheetError)) throw error;
        this.cache.set(name,{ok:false,code:error.code,message:error.message}); this.trace.push(name+' '+error.code);
        throw error;
      } finally { visiting.delete(name); }
    }
    _evaluate(node, visiting) {
      if (node.type==='number') return node.value;
      if (node.type==='ref') return this._resolve(node.name,visiting);
      if (node.type==='unary') { const value=this._evaluate(node.child,visiting); return node.op==='-' ? -value : value; }
      if (node.type==='binary') {
        const a=this._evaluate(node.left,visiting), b=this._evaluate(node.right,visiting);
        if (node.op==='/' && b===0) fail('#DIV/0!', 'Division by zero.');
        return node.op==='+' ? a+b : node.op==='-' ? a-b : node.op==='*' ? a*b : a/b;
      }
      if (node.type==='call') {
        const values=[];
        for (const arg of node.args) {
          if (arg.type==='range') for (const cell of arg.cells) values.push(this._resolve(cell,visiting));
          else values.push(this._evaluate(arg,visiting));
        }
        return node.name==='SUM' ? values.reduce((a,b)=>a+b,0) : node.name==='MIN' ? Math.min(...values) : Math.max(...values);
      }
      fail('#PARSE!', 'Unsupported expression.');
    }
  }
  const api={Spreadsheet,SheetError,parse,address};
  if (typeof module!=='undefined' && module.exports) module.exports=api;
  else root.SheetEngine=api;
})(typeof window!=='undefined' ? window : globalThis);
