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

// --- TypeScript Interfaces ---
interface SimulationDataPoint {
  timestamp: string;
  value: number;
}

interface Plots {
  decomposition: string;
  comparison: string;
}

function App() {
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2025-01-31");
  const [simulationData, setSimulationData] = useState<SimulationDataPoint[]>(
    [],
  );
  const [plots, setPlots] = useState<Plots | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRunSimulation = async () => {
    setLoading(true);
    setError(null);
    setSimulationData([]);
    setPlots(null);

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_API_BASE_URL}/simulations/water_temperature`,
        {
          start_date: startDate,
          end_date: endDate,
        },
      );
      setSimulationData(response.data.simulation_data);
      setPlots(response.data.plots);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || "An unknown error occurred.");
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
    link.setAttribute("download", `simulation_${startDate}_to_${endDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="container">
      <header>
        <h1>BISTEP Sensor Data Simulator</h1>
        <p>Generate and visualize synthetic time-series data.</p>
      </header>

      <div className="controls card">
        <h2>Configuration</h2>
        <div className="date-picker">
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
        </div>
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
          <div className="card">
            <div className="card-header">
              <h2>Simulation Results</h2>
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
                    name="Simulated Temperature"
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

