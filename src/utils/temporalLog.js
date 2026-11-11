// Buffers one simulation step and appends it through the dev server,
// which writes data/temporal/*.jsonl. In Node tests there is no fetch.

const queue = [];
let lastOk = null;

export function logRecord(record) {
  queue.push(record);
}

export function temporalLogOk() {
  return lastOk;
}

export function saveJsonFile(name, content) {
  if (typeof fetch !== 'function') return Promise.resolve(false);
  return fetch('/api/save-json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, content })
  }).then((response) => response.ok).catch(() => false);
}

export function flushTemporal() {
  if (queue.length === 0) return;
  if (typeof fetch !== 'function') {
    queue.length = 0;
    return;
  }
  const records = queue.splice(0, queue.length);
  fetch('/api/temporal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records })
  }).then((response) => {
    lastOk = response.ok;
  }).catch(() => {
    lastOk = false;
  });
}
