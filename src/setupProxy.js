const fs = require('fs');
const path = require('path');

// Create React App loads this file into the dev server. Each simulation step
// is appended as one JSON line under data/temporal/.
module.exports = function setupProxy(app) {
  const dir = path.join(__dirname, '..', 'data', 'temporal');
  fs.mkdirSync(dir, { recursive: true });

  app.post('/api/temporal', (req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
        const records = Array.isArray(body.records) ? body.records : [];
        const signals = [];
        const predictions = [];
        for (let i = 0; i < records.length; i++) {
          const line = JSON.stringify(records[i]);
          if (records[i].type === 'prediction') predictions.push(line);
          else signals.push(line);
        }
        if (signals.length > 0) {
          fs.appendFileSync(path.join(dir, 'signals.jsonl'), signals.join('\n') + '\n');
        }
        if (predictions.length > 0) {
          fs.appendFileSync(path.join(dir, 'predictions.jsonl'), predictions.join('\n') + '\n');
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, wrote: records.length }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false }));
      }
    });
  });

  app.post('/api/save-json', (req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
        const name = path.basename(String(body.name || ''));
        if (!/^[a-z0-9.-]+\.json$/.test(name)) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false }));
          return;
        }
        fs.writeFileSync(path.join(dir, name), JSON.stringify(body.content));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false }));
      }
    });
  });
};
