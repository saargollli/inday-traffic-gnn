// Temporal graph network: one graph convolution per step, then a GRU over the window.
// Each node is one signal. Weights are shared. Loss is mean squared error on
// the congestion those signals will have HORIZON steps later.

export const FEATURES = 10;
export const WINDOW = 8;
export const HORIZON = 40;

const HIDDEN = 12;

function zeros(n) {
  return new Array(n).fill(0);
}

function matrix(rows, cols, scale) {
  const s = scale == null ? Math.sqrt(2 / (rows + cols)) : scale;
  const m = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols);
    for (let c = 0; c < cols; c++) row[c] = (Math.random() * 2 - 1) * s;
    m[r] = row;
  }
  return m;
}

function cloneMatrix(m) {
  return m.map((row) => row.slice());
}

function sigmoid(x) {
  if (x > 12) return 1;
  if (x < -12) return 0;
  return 1 / (1 + Math.exp(-x));
}

function clip(value) {
  if (value > 1) return 1;
  if (value < -1) return -1;
  return value;
}

export class TemporalGNN {
  constructor(numNodes, neighbors) {
    this.n = numNodes;
    this.f = FEATURES;
    this.h = HIDDEN;
    this.neighbors = neighbors;

    this.Wg = matrix(FEATURES, HIDDEN);
    this.bg = zeros(HIDDEN);
    this.Wz = matrix(HIDDEN, HIDDEN, 0.08);
    this.Uz = matrix(HIDDEN, HIDDEN, 0.08);
    this.bz = zeros(HIDDEN);
    this.Wr = matrix(HIDDEN, HIDDEN, 0.08);
    this.Ur = matrix(HIDDEN, HIDDEN, 0.08);
    this.br = zeros(HIDDEN);
    this.Wn = matrix(HIDDEN, HIDDEN, 0.08);
    this.Un = matrix(HIDDEN, HIDDEN, 0.08);
    this.bn = zeros(HIDDEN);
    this.Wo = matrix(HIDDEN, 1);
    this.bo = [0];

    this.loss = 1;
    this._allocGrads();
  }

  _allocGrads() {
    const z2 = (m) => m.map((row) => row.map(() => 0));
    this.gWg = z2(this.Wg);
    this.gBg = zeros(HIDDEN);
    this.gWz = z2(this.Wz);
    this.gUz = z2(this.Uz);
    this.gBz = zeros(HIDDEN);
    this.gWr = z2(this.Wr);
    this.gUr = z2(this.Ur);
    this.gBr = zeros(HIDDEN);
    this.gWn = z2(this.Wn);
    this.gUn = z2(this.Un);
    this.gBn = zeros(HIDDEN);
    this.gWo = z2(this.Wo);
    this.gBo = [0];
  }

  _zeroGrads() {
    const clear = (m) => {
      for (let i = 0; i < m.length; i++) m[i] = 0;
    };
    const clear2 = (m) => {
      for (let i = 0; i < m.length; i++) clear(m[i]);
    };
    clear2(this.gWg);
    clear(this.gBg);
    clear2(this.gWz);
    clear2(this.gUz);
    clear(this.gBz);
    clear2(this.gWr);
    clear2(this.gUr);
    clear(this.gBr);
    clear2(this.gWn);
    clear2(this.gUn);
    clear(this.gBn);
    clear2(this.gWo);
    this.gBo[0] = 0;
  }

  graphConv(X) {
    const { n, f, h } = this;
    const S = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = X[i];
      const row = new Array(h);
      for (let k = 0; k < h; k++) {
        let sum = this.bg[k];
        for (let a = 0; a < f; a++) sum += x[a] * this.Wg[a][k];
        row[k] = sum;
      }
      S[i] = row;
    }

    const pre = new Array(n);
    const G = new Array(n);
    for (let i = 0; i < n; i++) {
      const nbrs = this.neighbors[i];
      const deg = nbrs.length;
      const p = new Array(h);
      const g = new Array(h);
      for (let k = 0; k < h; k++) {
        let acc = S[i][k];
        if (deg > 0) {
          let ns = 0;
          for (let t = 0; t < deg; t++) ns += S[nbrs[t]][k];
          acc += ns / deg;
        }
        p[k] = acc;
        g[k] = acc > 0 ? acc : 0;
      }
      pre[i] = p;
      G[i] = g;
    }
    return { pre, G };
  }

  gruStep(G, hPrev) {
    const { n, h } = this;
    const z = new Array(n);
    const r = new Array(n);
    const nHat = new Array(n);
    const rh = new Array(n);
    const hNew = new Array(n);

    for (let i = 0; i < n; i++) {
      const zi = new Array(h);
      const ri = new Array(h);
      const ni = new Array(h);
      const rhi = new Array(h);
      const hi = new Array(h);
      const gi = G[i];
      const hp = hPrev[i];

      for (let k = 0; k < h; k++) {
        let zp = this.bz[k];
        let rp = this.br[k];
        for (let j = 0; j < h; j++) {
          zp += gi[j] * this.Wz[j][k] + hp[j] * this.Uz[j][k];
          rp += gi[j] * this.Wr[j][k] + hp[j] * this.Ur[j][k];
        }
        const zv = sigmoid(zp);
        const rv = sigmoid(rp);
        zi[k] = zv;
        ri[k] = rv;
        rhi[k] = rv * hp[k];
      }
      for (let k = 0; k < h; k++) {
        let np = this.bn[k];
        for (let j = 0; j < h; j++) np += gi[j] * this.Wn[j][k] + rhi[j] * this.Un[j][k];
        const nv = Math.tanh(np);
        ni[k] = nv;
        hi[k] = (1 - zi[k]) * hp[k] + zi[k] * nv;
      }
      z[i] = zi;
      r[i] = ri;
      nHat[i] = ni;
      rh[i] = rhi;
      hNew[i] = hi;
    }
    return { z, r, n: nHat, rh, hNew, hPrev };
  }

  forward(window, store) {
    const { n, h } = this;
    let hidden = new Array(n);
    for (let i = 0; i < n; i++) hidden[i] = zeros(h);
    const steps = store ? new Array(window.length) : null;

    for (let t = 0; t < window.length; t++) {
      const hPrev = new Array(n);
      for (let i = 0; i < n; i++) hPrev[i] = hidden[i].slice();
      const conv = this.graphConv(window[t]);
      const gru = this.gruStep(conv.G, hPrev);
      hidden = gru.hNew;
      if (store) {
        steps[t] = { X: window[t], pre: conv.pre, G: conv.G, ...gru };
      }
    }

    const y = new Array(n);
    for (let i = 0; i < n; i++) {
      let logit = this.bo[0];
      const hi = hidden[i];
      for (let k = 0; k < h; k++) logit += hi[k] * this.Wo[k][0];
      y[i] = sigmoid(logit);
    }
    return { y, steps };
  }

  _gruBackward(cache, dH) {
    const { n, h } = this;
    const dG = new Array(n);
    const dHp = new Array(n);
    const { G, z, r, n: nHat, rh, hPrev } = cache;

    for (let i = 0; i < n; i++) {
      const dGi = zeros(h);
      const dHpi = zeros(h);
      const dzPre = new Array(h);
      const drPre = new Array(h);
      const dnPre = new Array(h);
      const drh = zeros(h);
      const zi = z[i];
      const ri = r[i];
      const ni = nHat[i];
      const gi = G[i];
      const hp = hPrev[i];
      const dhi = dH[i];

      for (let k = 0; k < h; k++) {
        const dh = dhi[k];
        const dz = (ni[k] - hp[k]) * dh;
        const dn = zi[k] * dh;
        dHpi[k] += (1 - zi[k]) * dh;
        dnPre[k] = dn * (1 - ni[k] * ni[k]);
        dzPre[k] = dz * zi[k] * (1 - zi[k]);
      }

      for (let k = 0; k < h; k++) {
        this.gBn[k] += dnPre[k];
        for (let j = 0; j < h; j++) {
          this.gWn[j][k] += gi[j] * dnPre[k];
          this.gUn[j][k] += rh[i][j] * dnPre[k];
          drh[j] += this.Un[j][k] * dnPre[k];
          dGi[j] += this.Wn[j][k] * dnPre[k];
        }
      }

      for (let k = 0; k < h; k++) {
        const dr = drh[k] * hp[k];
        dHpi[k] += drh[k] * ri[k];
        drPre[k] = dr * ri[k] * (1 - ri[k]);
      }

      for (let k = 0; k < h; k++) {
        this.gBz[k] += dzPre[k];
        this.gBr[k] += drPre[k];
        for (let j = 0; j < h; j++) {
          this.gWz[j][k] += gi[j] * dzPre[k];
          this.gUz[j][k] += hp[j] * dzPre[k];
          this.gWr[j][k] += gi[j] * drPre[k];
          this.gUr[j][k] += hp[j] * drPre[k];
          dGi[j] += this.Wz[j][k] * dzPre[k] + this.Wr[j][k] * drPre[k];
          dHpi[j] += this.Uz[j][k] * dzPre[k] + this.Ur[j][k] * drPre[k];
        }
      }

      dG[i] = dGi;
      dHp[i] = dHpi;
    }
    return { dG, dHp };
  }

  _graphBackward(X, pre, dG) {
    const { n, f, h } = this;
    const dPre = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(h);
      const p = pre[i];
      const g = dG[i];
      for (let k = 0; k < h; k++) row[k] = p[k] > 0 ? g[k] : 0;
      dPre[i] = row;
    }

    const dS = new Array(n);
    for (let i = 0; i < n; i++) dS[i] = zeros(h);

    for (let i = 0; i < n; i++) {
      const nbrs = this.neighbors[i];
      const deg = nbrs.length;
      const dp = dPre[i];
      for (let k = 0; k < h; k++) {
        dS[i][k] += dp[k];
        if (deg > 0) {
          const share = dp[k] / deg;
          for (let t = 0; t < deg; t++) dS[nbrs[t]][k] += share;
        }
      }
    }

    for (let i = 0; i < n; i++) {
      const x = X[i];
      const ds = dS[i];
      for (let k = 0; k < h; k++) this.gBg[k] += ds[k];
      for (let a = 0; a < f; a++) {
        const xv = x[a];
        if (xv === 0) continue;
        const row = this.gWg[a];
        for (let k = 0; k < h; k++) row[k] += xv * ds[k];
      }
    }
  }

  accumulate(window, targets, nodeIds) {
    this._zeroGrads();
    const { y, steps } = this.forward(window, true);
    const count = nodeIds.length;
    let loss = 0;
    let weightSum = 0;
    const dLogit = zeros(this.n);
    for (let s = 0; s < count; s++) {
      const i = nodeIds[s];
      const weight = 1 + 4 * targets[i];
      weightSum += weight;
      const err = y[i] - targets[i];
      loss += weight * err * err;
      dLogit[i] = weight * err;
    }
    loss /= weightSum;
    for (let s = 0; s < count; s++) {
      const i = nodeIds[s];
      dLogit[i] = (2 * dLogit[i] / weightSum) * y[i] * (1 - y[i]);
    }

    const hLast = steps[steps.length - 1].hNew;
    let dH = new Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const row = zeros(this.h);
      const dl = dLogit[i];
      if (dl !== 0) {
        this.gBo[0] += dl;
        for (let k = 0; k < this.h; k++) {
          this.gWo[k][0] += hLast[i][k] * dl;
          row[k] = this.Wo[k][0] * dl;
        }
      }
      dH[i] = row;
    }

    for (let t = steps.length - 1; t >= 0; t--) {
      const back = this._gruBackward(steps[t], dH);
      this._graphBackward(steps[t].X, steps[t].pre, back.dG);
      dH = back.dHp;
    }

    this.loss = loss;
    return loss;
  }

  apply(lr) {
    const step2 = (W, G) => {
      for (let i = 0; i < W.length; i++) {
        const row = W[i];
        const g = G[i];
        for (let j = 0; j < row.length; j++) row[j] -= lr * clip(g[j]);
      }
    };
    const step1 = (b, g) => {
      for (let i = 0; i < b.length; i++) b[i] -= lr * clip(g[i]);
    };
    step2(this.Wg, this.gWg);
    step1(this.bg, this.gBg);
    step2(this.Wz, this.gWz);
    step2(this.Uz, this.gUz);
    step1(this.bz, this.gBz);
    step2(this.Wr, this.gWr);
    step2(this.Ur, this.gUr);
    step1(this.br, this.gBr);
    step2(this.Wn, this.gWn);
    step2(this.Un, this.gUn);
    step1(this.bn, this.gBn);
    step2(this.Wo, this.gWo);
    step1(this.bo, this.gBo);
  }

  train(window, targets, nodeIds, lr) {
    const loss = this.accumulate(window, targets, nodeIds);
    this.apply(lr);
    return loss;
  }

  getWeights() {
    return {
      Wg: cloneMatrix(this.Wg), bg: this.bg.slice(),
      Wz: cloneMatrix(this.Wz), Uz: cloneMatrix(this.Uz), bz: this.bz.slice(),
      Wr: cloneMatrix(this.Wr), Ur: cloneMatrix(this.Ur), br: this.br.slice(),
      Wn: cloneMatrix(this.Wn), Un: cloneMatrix(this.Un), bn: this.bn.slice(),
      Wo: cloneMatrix(this.Wo), bo: this.bo.slice()
    };
  }

  setWeights(w) {
    this.Wg = cloneMatrix(w.Wg); this.bg = w.bg.slice();
    this.Wz = cloneMatrix(w.Wz); this.Uz = cloneMatrix(w.Uz); this.bz = w.bz.slice();
    this.Wr = cloneMatrix(w.Wr); this.Ur = cloneMatrix(w.Ur); this.br = w.br.slice();
    this.Wn = cloneMatrix(w.Wn); this.Un = cloneMatrix(w.Un); this.bn = w.bn.slice();
    this.Wo = cloneMatrix(w.Wo); this.bo = w.bo.slice();
  }

  // Each intersection trains one step from the same start weights. The server
  // then keeps the average of those client weights (FedAvg).
  federatedRound(window, targets, groups, lr) {
    const start = this.getWeights();
    const updated = [];
    const masses = [];
    let loss = 0;
    let lossWeight = 0;
    for (let g = 0; g < groups.length; g++) {
      const nodes = groups[g];
      if (nodes.length === 0) continue;
      let mass = 0.2;
      for (let n = 0; n < nodes.length; n++) mass += targets[nodes[n]] || 0;
      this.setWeights(start);
      const local = this.train(window, targets, nodes, lr);
      updated.push(this.getWeights());
      masses.push(mass);
      loss += local * mass;
      lossWeight += mass;
    }
    if (updated.length === 0) return this.loss;
    this.setWeights(averageWeights(updated, masses));
    this.loss = lossWeight > 0 ? loss / lossWeight : this.loss;
    return this.loss;
  }
}

function averageWeights(list, masses) {
  const acc = list[0];
  const n = list.length;
  let total = 0;
  for (let i = 0; i < n; i++) total += masses ? masses[i] : 1;
  if (total === 0) total = n;
  const keys2 = ['Wg', 'Wz', 'Uz', 'Wr', 'Ur', 'Wn', 'Un', 'Wo'];
  const keys1 = ['bg', 'bz', 'br', 'bn', 'bo'];
  const out = {
    Wg: acc.Wg.map((row) => row.map(() => 0)),
    Wz: acc.Wz.map((row) => row.map(() => 0)),
    Uz: acc.Uz.map((row) => row.map(() => 0)),
    Wr: acc.Wr.map((row) => row.map(() => 0)),
    Ur: acc.Ur.map((row) => row.map(() => 0)),
    Wn: acc.Wn.map((row) => row.map(() => 0)),
    Un: acc.Un.map((row) => row.map(() => 0)),
    Wo: acc.Wo.map((row) => row.map(() => 0)),
    bg: acc.bg.map(() => 0),
    bz: acc.bz.map(() => 0),
    br: acc.br.map(() => 0),
    bn: acc.bn.map(() => 0),
    bo: acc.bo.map(() => 0)
  };
  for (let i = 0; i < n; i++) {
    const w = list[i];
    for (let k = 0; k < keys2.length; k++) {
      const key = keys2[k];
      const A = out[key];
      const B = w[key];
      for (let r = 0; r < A.length; r++) {
        for (let c = 0; c < A[r].length; c++) A[r][c] += B[r][c] * ((masses ? masses[i] : 1) / total);
      }
    }
    for (let k = 0; k < keys1.length; k++) {
      const key = keys1[k];
      const A = out[key];
      const B = w[key];
      for (let r = 0; r < A.length; r++) A[r] += B[r] * ((masses ? masses[i] : 1) / total);
    }
  }
  return out;
}

export function mse(model, window, targets, nodeIds) {
  const { y } = model.forward(window, false);
  let sum = 0;
  for (let s = 0; s < nodeIds.length; s++) {
    const err = y[nodeIds[s]] - targets[nodeIds[s]];
    sum += err * err;
  }
  return sum / nodeIds.length;
}
