'use strict';
const { Spreadsheet } = require('./engine.js');
const sheet = new Spreadsheet('lazy');
sheet.set('A1', '10');
sheet.set('B1', '=A1 * 2');
sheet.set('C1', '=SUM(A1:B1)');
console.log('Initial C1:', sheet.get('C1')); // 30
sheet.set('A1', '15');
console.log('Affected:', sheet.lastEdit.affected);
console.log('Computed during lazy write:', sheet.lastEdit.computed); // []
console.log('Updated C1:', sheet.get('C1')); // 45
console.log('Read trace:', sheet.trace);
try { sheet.set('A1', '=C1'); }
catch (error) { console.log('Rejected:', error.code, error.message); }
console.log('A1 retained:', sheet.raw('A1')); // 15
sheet.setMode('eager');
sheet.set('A1', '20');
console.log('Computed during eager write:', sheet.lastEdit.computed);
console.log('Final C1:', sheet.get('C1')); // 60
