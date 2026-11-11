import { useEffect, useRef, useState } from 'react';
import CanvasRenderer from './components/CanvasRenderer';
import { createSimulation, TICK_MS } from './simulation/engine';
import { runInitialDatasetTest, runSameDatasetTest } from './simulation/experiment';
import { TRAFFIC_LEVELS } from './utils/trafficColor';
import './App.css';

function compareText(result) {
  const mapDelta = (result.trained.mapLoad - result.untrained.mapLoad) * 100;
  const peakDelta = (result.trained.peakLoad - result.untrained.peakLoad) * 100;
  const map = Math.abs(mapDelta) < 0.05
    ? 'Average traffic across the map stayed about the same.'
    : `Average traffic across the map ${mapDelta < 0 ? 'fell' : 'rose'} by ${Math.abs(mapDelta).toFixed(1)} points.`;
  const peak = Math.abs(peakDelta) < 0.05
    ? 'The busiest road stayed about the same.'
    : `The busiest road ${peakDelta < 0 ? 'fell' : 'rose'} by ${Math.abs(peakDelta).toFixed(1)} points.`;
  return `${map} ${peak} Trained for ${result.rounds} rounds, loss ${result.loss.toFixed(3)}.`;
}

function App() {
  const simRef = useRef(null);
  if (!simRef.current) simRef.current = createSimulation();

  const [running, setRunning] = useState(false);
  const [panel, setPanel] = useState(() => simRef.current.panel());
  const [lastAdd, setLastAdd] = useState(null);
  const [tripCount, setTripCount] = useState(12);
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
        setPanel(simRef.current.panel());
      } finally {
        busy = false;
      }
    }, TICK_MS);
    return () => clearInterval(id);
  }, [running]);

  const trained = panel.rounds > 0;
  const currentStep = !compare || !compare.untrained ? 1 : !compare.trained ? 2 : 3;
  const steps = [
    { n: 1, title: 'Initial test', detail: 'Run the prepared trips on the untrained model.' },
    { n: 2, title: 'Collect', detail: 'Add vehicles, then Run. The model trains on those cars.' },
    { n: 3, title: 'Test again', detail: 'Run the same trips. The trained model names a lighter path.' }
  ];
  const roadsAreEmpty = panel.busiest.every((light) => light.level === 'empty');

  return (
    <div className="traffic-app">
      <header className="traffic-header">
        <div>
          <p className="traffic-kicker">Temporal graph model on each signal</p>
          <h1>Predicting Traffic Congestion Using Federated Learning</h1>
      </div>
        <div className="traffic-status">
          <strong>{running ? 'Running' : 'Paused'} · {panel.mode === 'test' ? 'Test' : 'Collect'}</strong>
          {panel.conflicts === 0 ? 'One green per intersection' : `${panel.conflicts} intersections have two greens`}
        </div>
      </header>

      <main className="traffic-main">
        <section className="net-card">
          <ol className="steps">
            {steps.map((step) => {
              const state = step.n === currentStep ? 'current' : step.n < currentStep ? 'done' : 'waiting';
              return (
                <li key={step.n} className={`step ${state}`}>
                  <b><span className="step-num">{step.n < currentStep ? '✓' : step.n}</span>{step.title}</b>
                  <span>{step.detail}</span>
                </li>
              );
            })}
          </ol>
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
                  const added = simRef.current.addVehicles();
                  setLastAdd(added);
                  setPanel(simRef.current.panel());
                }}
              >
                Add vehicles
                        </button>
                        <button
                className="btn-reset"
                onClick={() => {
                  setRunning(false);
                  simRef.current.resetEnvironment();
                  setLastAdd(null);
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
            <p className="file-note">Leave Collect selected. Add vehicles and Run is what trains the model.</p>
            {lastAdd && (
              <p className="file-note">
                Last add: {lastAdd.added} vehicles, {lastAdd.main} on a main corridor, {lastAdd.random} choosing randomly at each intersection.
              </p>
            )}
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
              <span style={{ marginLeft: 8 }}>Number on a moving car is its speed.</span>
            </p>
          </section>

          <section className="panel-card">
            <h2>Fixed-trip test</h2>
            <ol className="mini-steps">
              <li>Test the prepared source and destination list on the initial model.</li>
              <li>Press Add vehicles, then Run. Wait until FedAvg rounds is above 0.</li>
              <li>Test that same list again and compare the two results.</li>
            </ol>
            <label className="trip-count">
              Vehicles
              <input
                type="number"
                min="4"
                max="20"
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
                      onProgress: (note) => {
                        setCompareNote(note);
                        setPanel(simRef.current.panel());
                      }
                    });
                    setCompare(result);
                    setCompareNote(result.saved
                      ? 'Initial result saved. Now press Add vehicles, then Run.'
                      : 'Initial result is shown. Now press Add vehicles, then Run.');
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
                      onProgress: (note) => {
                        setCompareNote(note);
                        setPanel(simRef.current.panel());
                      }
                    });
                    setCompare(result);
                    setCompareNote(result.saved
                      ? 'Saved under data/temporal.'
                      : 'Shown here. Restart npm start to write the files.');
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
                      <td>Finished</td>
                      <td>{compare.untrained.finished}/{compare.dataset.count}</td>
                      <td>{compare.trained ? `${compare.trained.finished}/${compare.dataset.count}` : '—'}</td>
                    </tr>
                  </tbody>
                </table>
                {compare.trained && (
                  <p className={compare.trained.peakLoad < compare.untrained.peakLoad ? 'file-note save-ok' : 'file-note'}>
                    {compareText(compare)}
                  </p>
                )}
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
              <div><span>Vehicles</span><strong>{panel.vehicles}</strong></div>
              <div><span>Step</span><strong>{panel.step}</strong></div>
              <div><span>Samples saved</span><strong>{panel.samples}</strong></div>
              <div><span>FedAvg rounds</span><strong>{panel.rounds}</strong></div>
              <div><span>Train loss</span><strong>{panel.loss.toFixed(3)}</strong></div>
              <div><span>Forecast fit</span><strong>{panel.accuracy == null ? '—' : `${panel.accuracy.toFixed(0)}%`}</strong></div>
                    </div>
            <p className="file-note">
              Greens now: {panel.greens}. Yellow transitions: {panel.yellows}. Each green length is drawn separately, so the signals do not flip together.
            </p>
            <p className={panel.saving === false ? 'file-note save-bad' : 'file-note save-ok'}>
              {panel.saving === false
                ? 'Could not write data/temporal. Restart npm start so the logger is loaded.'
                : 'Writing data/temporal/signals.jsonl and predictions.jsonl'}
            </p>
          </section>

          <section className="panel-card">
            <h2>Busiest signals</h2>
            {roadsAreEmpty ? (
              <p className="file-note">No cars on the roads yet. A road with cars turns green, then amber, then red as it fills.</p>
            ) : panel.busiest.filter((light) => light.level !== 'empty').map((light) => (
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
                {panel.notices.map((notice) => (
                  <div className="notice-row" key={`${notice.id}-${notice.step}`}>
                    <span>Vehicle {notice.id}</span>
                    <span>{notice.text}</span>
                        </div>
                ))}
                      </>
                    )}
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;
