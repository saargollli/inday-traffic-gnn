import { useEffect, useRef, useState } from 'react';
import CanvasRenderer from './components/CanvasRenderer';
import { createSimulation, TICK_MS } from './simulation/engine';
import { buildTrainingDataset, clampTripCount, runInitialDatasetTest, runSameDatasetTest } from './simulation/experiment';
import { TRAFFIC_LEVELS } from './utils/trafficColor';
import './App.css';

function App() {
  const simRef = useRef(null);
  if (!simRef.current) simRef.current = createSimulation();

  const [running, setRunning] = useState(false);
  const [panel, setPanel] = useState(() => simRef.current.panel());
  const [tripCount, setTripCount] = useState(24);
  const [compareBusy, setCompareBusy] = useState(false);
  const [compareNote, setCompareNote] = useState('');
  const [compare, setCompare] = useState(null);

  useEffect(() => {
    if (!running) return undefined;
    let busy = false;
    const id = setInterval(() => {
      if (busy) return;
      busy = true;
      try {
        simRef.current.advance();
        const next = simRef.current.panel();
        setPanel(next);
        if (next.cohortTotal > 0 && next.vehicles === 0 && next.cohortDone >= next.cohortTotal) {
          setRunning(false);
          simRef.current.fitLearnedTrips();
          setPanel(simRef.current.panel());
        }
    } finally {
        busy = false;
      }
    }, TICK_MS);
    return () => clearInterval(id);
  }, [running]);

  const trained = panel.rounds > 0;

  return (
    <div className="traffic-app">
      <header className="traffic-header">
        <h1>Predicting Traffic Congestion Using Federated Learning</h1>
        <div className="traffic-status">
          <strong>{running ? 'Running' : 'Paused'} · {panel.mode === 'test' ? 'Test' : 'Collect'}</strong>
        </div>
      </header>

      <main className="traffic-main">
        <section className="net-card">
          <CanvasRenderer simRef={simRef} />
        </section>

        <aside className="side-panel">
          <section className="panel-card">
            <h2>Run</h2>
            <div className="controls">
                        <button
                className={running ? 'btn-pause' : 'btn-run'}
                onClick={() => setRunning((value) => !value)}
                        >
                {running ? 'Pause' : 'Run'}
                        </button>
                        <button
                className="btn-add"
                onClick={() => {
                  const sim = simRef.current;
                  const count = clampTripCount(tripCount);
                  sim.cohortSeed = (sim.cohortSeed || 0) + 1;
                  const batch = buildTrainingDataset(sim, count, Date.now() + sim.cohortSeed * 97);
                  sim.addTripBatch(batch);
                  setPanel(sim.panel());
                }}
              >
                Add vehicles
                        </button>
                        <button
                className="btn-reset"
                onClick={() => {
                  setRunning(false);
                  simRef.current.resetEnvironment();
                  setPanel(simRef.current.panel());
                }}
              >
                Reset environment
                        </button>
            </div>
            <div className="mode-switch">
                        <button
                className={panel.mode === 'collect' ? 'active' : ''}
                onClick={() => {
                  simRef.current.setMode('collect');
                  setPanel(simRef.current.panel());
                }}
              >
                Collect
                        </button>
                        <button
                className={panel.mode === 'test' ? 'active' : ''}
                disabled={!trained}
                onClick={() => {
                  simRef.current.setMode('test');
                  setPanel(simRef.current.panel());
                }}
              >
                Test model
                        </button>
                    </div>
            <p className="legend">
              {TRAFFIC_LEVELS.map((level) => (
                <span key={level.id} className="legend-item">
                  <span className="swatch" style={{ background: level.color }} /> {level.label}
                                </span>
              ))}
            </p>
            <p className="legend">
              <span className="swatch" style={{ background: '#1d4ed8' }} /> Main corridor
              <span className="swatch" style={{ background: '#64748b', marginLeft: 8 }} /> Random
              <span className="swatch" style={{ background: '#ea580c', marginLeft: 8 }} /> Rerouted
            </p>
          </section>

          <section className="panel-card">
            <h2>Fixed-trip test</h2>
            <label className="trip-count">
              Vehicles (4–60)
              <input
                type="number"
                min="4"
                max="60"
                value={tripCount}
                disabled={compareBusy}
                onChange={(event) => setTripCount(Number(event.target.value))}
              />
            </label>
            <div className="compare-actions">
                      <button
                className="btn-add compare-btn"
                disabled={compareBusy}
                onClick={async () => {
                  setRunning(false);
                  setCompareBusy(true);
                  setCompare(null);
                  try {
                    const result = await runInitialDatasetTest(simRef.current, {
                      count: tripCount,
                      onProgress: () => setPanel(simRef.current.panel())
                    });
                    setCompare(result);
                    setCompareNote('');
                  } catch (err) {
                    setCompareNote(err && err.message ? err.message : 'Initial test failed');
                  } finally {
                    setCompareBusy(false);
                    setPanel(simRef.current.panel());
                  }
                }}
              >
                {compareBusy && !compare ? 'Running the dataset' : 'Test initial model'}
                      </button>
                      <button
                className="btn-run compare-btn"
                disabled={compareBusy || !compare || !compare.untrained || !trained}
                onClick={async () => {
                  setRunning(false);
                  setCompareBusy(true);
                  try {
                    const result = await runSameDatasetTest(simRef.current, {
                      onProgress: () => setPanel(simRef.current.panel())
                    });
                    setCompare(result);
                    setCompareNote('');
                  } catch (err) {
                    setCompareNote(err && err.message ? err.message : 'Dataset test failed');
                  } finally {
                    setCompareBusy(false);
                    setPanel(simRef.current.panel());
                  }
                }}
              >
                {compareBusy && compare ? 'Running the dataset' : 'Test the same dataset'}
                      </button>
                    </div>
            {compareNote && <p className="file-note">{compareNote}</p>}
            {compare && (
              <>
                <table className="compare-table">
                  <thead>
                    <tr><th></th><th>Untrained</th><th>Trained</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Map traffic</td>
                      <td>{(compare.untrained.mapLoad * 100).toFixed(1)}%</td>
                      <td>{compare.trained ? `${(compare.trained.mapLoad * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                    <tr>
                      <td>Peak road</td>
                      <td>{(compare.untrained.peakLoad * 100).toFixed(1)}%</td>
                      <td>{compare.trained ? `${(compare.trained.peakLoad * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                    <tr>
                      <td>Waiting</td>
                      <td>{compare.untrained.meanWaiting.toFixed(2)}</td>
                      <td>{compare.trained ? compare.trained.meanWaiting.toFixed(2) : '—'}</td>
                    </tr>
                    <tr>
                      <td>Finished all trips</td>
                      <td>{compare.untrained.finished}/{compare.dataset.count}</td>
                      <td>{compare.trained ? `${compare.trained.finished}/${compare.dataset.count}` : '—'}</td>
                    </tr>
                  </tbody>
                </table>
                <div className="trip-list">
                  {compare.dataset.vehicles.slice(0, 6).map((vehicle) => {
                    const before = compare.untrained.vehicles.find((row) => row.id === vehicle.id);
                    const after = compare.trained
                      ? compare.trained.vehicles.find((row) => row.id === vehicle.id)
                      : null;
                    return (
                      <div key={vehicle.id} className="trip-plan">
                        <b>Vehicle {vehicle.id}</b>
                        <span>{vehicle.plan}</span>
                        <span>Untrained: {before ? before.path : '—'}</span>
                        <span>Trained: {compare.trained && after ? after.path : '—'}</span>
                    </div>
                    );
                  })}
                  </div>
              </>
            )}
          </section>

          <section className="panel-card">
            <h2>Series</h2>
            <div className="stat-grid">
              <div><span>Still driving</span><strong>{panel.vehicles}</strong></div>
              <div><span>Finished trips</span><strong>{panel.cohortTotal ? `${panel.cohortDone}/${panel.cohortTotal}` : '—'}</strong></div>
              <div><span>Step</span><strong>{panel.step}</strong></div>
              <div><span>Samples saved</span><strong>{panel.samples}</strong></div>
              <div><span>FedAvg rounds</span><strong>{panel.rounds}</strong></div>
              <div><span>Train loss</span><strong>{panel.loss.toFixed(3)}</strong></div>
              <div><span>Forecast fit</span><strong>{panel.accuracy == null ? '—' : `${panel.accuracy.toFixed(0)}%`}</strong></div>
                      </div>
          </section>

          <section className="panel-card">
            <h2>Busiest signals</h2>
            {panel.busiest.filter((light) => light.level !== 'empty').map((light) => (
              <div
                className="light-row"
                key={light.id}
                style={{ background: light.tint, borderLeftColor: light.color }}
              >
                <div className="light-name">
                  {light.name}
                  <small>
                    from {light.from} · {light.vehicles} {light.vehicles === 1 ? 'car' : 'cars'} · {light.levelLabel}
                    {panel.mode === 'test' && light.prediction != null ? ` · forecast ${light.prediction.toFixed(2)}` : ''}
                  </small>
                    </div>
                <span className={`pill ${light.state}`}>{light.state}</span>
                  </div>
            ))}
            {panel.notices.length > 0 && (
              <>
                <h2 style={{ marginTop: 12 }}>Notices</h2>
                {panel.notices.map((notice) => {
                  const lines = notice.lines && notice.lines.length ? notice.lines : [notice.text];
                  return (
                    <div className="notice-row" key={`${notice.id}-${notice.step}`}>
                      <span>Vehicle {notice.id}</span>
                      <span className="notice-copy">
                        {lines.map((line) => (
                          <span
                            key={line}
                            className={line.indexOf('Faster route:') === 0 ? 'faster' : line.indexOf('Destination:') === 0 ? 'dest' : ''}
                          >
                            {line}
                    </span>
                        ))}
                    </span>
                  </div>
                  );
                })}
                      </>
                    )}
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;
