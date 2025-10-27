import { useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import Papa from "papaparse";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";

// --- TypeScript Interfaces ---
interface SimulationDataPoint {
  timestamp: string;
  value: number;
}

interface Plots {
  decomposition: string;
  comparison: string;
}

// --- Constants ---
const intervalOptions = [
  { value: "1h", label: "1 Hour" },
  { value: "12h", label: "12 Hours" },
  { value: "1d", label: "1 Day" },
  { value: "3d", label: "3 Days" },
  { value: "7d", label: "7 Days" },
  { value: "30d", label: "30 Days" },
];

function App() {
  const [sensorType, setSensorType] = useState("water_temperature");
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2025-01-31");
  const [interval, setInterval] = useState(intervalOptions[2].value); // Default to 1 Day
  const [simulationData, setSimulationData] = useState<SimulationDataPoint[]>(
    [],
  );
  const [plots, setPlots] = useState<Plots | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formula, setFormula] = useState<string | null>(null);

  // Anode Lifetime specific parameters
  const [anodeConstantDuration, setAnodeConstantDuration] = useState(24 * 30 * 6); // 6 months in hours
  const [anodeDecayRate, setAnodeDecayRate] = useState(0.001);
  const [anodeNoiseLevel, setAnodeNoiseLevel] = useState(0.01);

  const handleRunSimulation = async () => {
    setLoading(true);
    setError(null);
    setSimulationData([]);
    setPlots(null);
    setFormula(null);

    try {
      let response;
      if (sensorType === "anode_lifetime") {
        response = await axios.post(
          `${import.meta.env.VITE_API_BASE_URL}/simulations/anode-lifetime`,
          {
            start_date: startDate,
            end_date: endDate,
            interval: interval,
            constant_duration: anodeConstantDuration,
            decay_rate: anodeDecayRate,
            noise_level: anodeNoiseLevel,
          },
        );
        setSimulationData(response.data.simulation_data);
        setFormula(response.data.formula);
        setPlots(null); // Anode lifetime does not have plots
      } else {
        response = await axios.post(
          `${import.meta.env.VITE_API_BASE_URL}/simulations/${sensorType}`,
          {
            start_date: startDate,
            end_date: endDate,
            interval: interval,
          },
        );
        setSimulationData(response.data.simulation_data);
        setPlots(response.data.plots);
      }
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        const detail = err.response.data.detail;
        if (typeof detail === "string") {
          setError(detail);
        } else if (Array.isArray(detail)) {
          const errorString = detail
            .map((d) => `[${d.loc.join(" -> ")}]: ${d.msg}`)
            .join("; ");
          setError(errorString);
        } else {
          setError(JSON.stringify(err.response.data));
        }
      } else {
        setError("An unexpected error occurred.");
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    const csv = Papa.unparse(simulationData);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `${sensorType}_${interval}_simulation_${startDate}_to_${endDate}.csv`,
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getSensorDisplayName = (sensor: string) => {
    if (sensor === "water_temperature") return "Water Temperature";
    if (sensor === "tidal_level") return "Tidal Level";
    if (sensor === "anode_lifetime") return "Anode Lifetime";
    return "Salinity";
  };

  return (
    <div className="container">
      <header>
        <h1>BISTEP Sensor Data Simulator</h1>
        <p>Generate and visualize synthetic time-series data.</p>
      </header>
      <div className="controls card">
        <h2>Configuration</h2>
        <div className="control-grid">
          <div>
            <label htmlFor="sensor-type">Sensor Type</label>
            <select
              id="sensor-type"
              value={sensorType}
              onChange={(e) => setSensorType(e.target.value)}
            >
              <option value="water_temperature">Water Temperature</option>
              <option value="salinity">Salinity</option>
              <option value="tidal_level">Tidal Level</option>
              <option value="anode_lifetime">Anode Lifetime</option>
            </select>
          </div>
          <div>
            <label htmlFor="start-date">Start Date</label>
            <input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="end-date">End Date</label>
            <input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="interval">Interval</label>
            <select
              id="interval"
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
            >
              {intervalOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {sensorType === "anode_lifetime" && (
          <div className="anode-controls">
            <h3>Anode Lifetime Parameters</h3>
            <div className="control-grid">
              <div>
                <label htmlFor="constant-duration">
                  Constant Duration (months):{" "}
                  {Math.round(anodeConstantDuration / (24 * 30))}
                </label>
                <input
                  id="constant-duration"
                  type="range"
                  min="0"
                  max={24 * 365 * 3} // 3 years in hours
                  step={24 * 30} // 1 month steps
                  value={anodeConstantDuration}
                  onChange={(e) =>
                    setAnodeConstantDuration(Number(e.target.value))
                  }
                />
              </div>
              <div>
                <label htmlFor="decay-rate">Decay Rate: {anodeDecayRate}</label>
                <input
                  id="decay-rate"
                  type="range"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={anodeDecayRate}
                  onChange={(e) => setAnodeDecayRate(Number(e.target.value))}
                />
              </div>
              <div>
                <label htmlFor="noise-level">Noise Level: {anodeNoiseLevel}</label>
                <input
                  id="noise-level"
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.005"
                  value={anodeNoiseLevel}
                  onChange={(e) => setAnodeNoiseLevel(Number(e.target.value))}
                />
              </div>
            </div>
          </div>
        )}

        <button onClick={handleRunSimulation} disabled={loading}>
          {loading ? "Generating..." : "Run Simulation"}
        </button>
      </div>
      {error && (
        <div className="card error-card">
          <strong>Error:</strong> {error}
        </div>
      )}
      {loading && (
        <div className="card loading-card">
          Loading data and generating plots...
        </div>
      )}
      {simulationData.length > 0 && (
        <div className="results-container">
          {formula && (
            <div className="card">
              <h2>Simulation Formula</h2>
              <BlockMath math={formula} />
            </div>
          )}
          <div className="card">
            <div className="card-header">
              <h2>{getSensorDisplayName(sensorType)} Simulation Results</h2>
              <button onClick={handleDownloadCSV}>Download CSV</button>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={simulationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(ts) => new Date(ts).toLocaleDateString()}
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    name={`Simulated ${getSensorDisplayName(sensorType)}`}
                    stroke="#8884d8"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {plots && (
            <>
              <div className="card">
                <h2>Comparison Analysis</h2>
                <img
                  src={plots.comparison}
                  alt="Comparison of original vs. simulated data"
                />
              </div>
              <div className="card">
                <h2>Time Series Decomposition</h2>
                <img
                  src={plots.decomposition}
                  alt="Time series decomposition plot"
                />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
