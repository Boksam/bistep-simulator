import { useState } from "react";
import { useTranslation } from "react-i18next";
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
import LanguageSwitcher from "./components/LanguageSwitcher";

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
  { value: "1h", key: "1h" },
  { value: "12h", key: "12h" },
  { value: "1d", key: "1d" },
  { value: "3d", key: "3d" },
  { value: "7d", key: "7d" },
  { value: "30d", key: "30d" },
];

function App() {
  const { t } = useTranslation();
  const [sensorType, setSensorType] = useState("water_temperature");
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2025-01-31");
  const [interval, setInterval] = useState(intervalOptions[2].value); // Default to 1 Day
  const [simulationData, setSimulationData] = useState<SimulationDataPoint[]>(
    []
  );
  const [plots, setPlots] = useState<Plots | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formula, setFormula] = useState<string | null>(null);

  // Anode Lifetime specific parameters
  const [anodeConstantDuration, setAnodeConstantDuration] = useState(
    24 * 30 * 6
  ); // 6 months in hours
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
          }
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
          }
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
        setError(t("messages.unexpectedError"));
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
      `${sensorType}_${interval}_simulation_${startDate}_to_${endDate}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getSensorDisplayName = (sensor: string) => {
    return t(`sensorTypes.${sensor}`);
  };

  return (
    <div className="container">
      <header>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <h1>{t("app.title")}</h1>
            <p>{t("app.description")}</p>
          </div>
          <LanguageSwitcher />
        </div>
      </header>
      <div className="controls card">
        <h2>{t("configuration.title")}</h2>
        <div className="control-grid">
          <div>
            <label htmlFor="sensor-type">{t("configuration.sensorType")}</label>
            <select
              id="sensor-type"
              value={sensorType}
              onChange={(e) => setSensorType(e.target.value)}
            >
              <option value="water_temperature">
                {t("sensorTypes.water_temperature")}
              </option>
              <option value="salinity">{t("sensorTypes.salinity")}</option>
              <option value="tidal_level">
                {t("sensorTypes.tidal_level")}
              </option>
              <option value="anode_lifetime">
                {t("sensorTypes.anode_lifetime")}
              </option>
            </select>
          </div>
          <div>
            <label htmlFor="start-date">{t("configuration.startDate")}</label>
            <input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="end-date">{t("configuration.endDate")}</label>
            <input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="interval">{t("configuration.interval")}</label>
            <select
              id="interval"
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
            >
              {intervalOptions.map((option) => (
                <option key={option.key} value={option.value}>
                  {t(`intervals.${option.key}`)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {sensorType === "anode_lifetime" && (
          <div className="anode-controls">
            <h3>{t("anodeLifetime.title")}</h3>
            <div className="control-grid">
              <div>
                <label htmlFor="constant-duration">
                  {t("anodeLifetime.constantDuration")}:{" "}
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
                <label htmlFor="decay-rate">
                  {t("anodeLifetime.decayRate")}: {anodeDecayRate}
                </label>
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
                <label htmlFor="noise-level">
                  {t("anodeLifetime.noiseLevel")}: {anodeNoiseLevel}
                </label>
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
          {loading ? t("buttons.generating") : t("buttons.runSimulation")}
        </button>
      </div>
      {error && (
        <div className="card error-card">
          <strong>{t("messages.error")}:</strong> {error}
        </div>
      )}
      {loading && (
        <div className="card loading-card">{t("messages.loading")}</div>
      )}
      {simulationData.length > 0 && (
        <div className="results-container">
          {formula && (
            <div className="card">
              <h2>{t("results.simulationFormula")}</h2>
              <BlockMath math={formula} />
            </div>
          )}
          <div className="card">
            <div className="card-header">
              <h2>
                {getSensorDisplayName(sensorType)}{" "}
                {t("results.simulationResults")}
              </h2>
              <button onClick={handleDownloadCSV}>
                {t("buttons.downloadCSV")}
              </button>
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
                    name={`${t("results.simulated")} ${getSensorDisplayName(
                      sensorType
                    )}`}
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
                <h2>{t("results.comparisonAnalysis")}</h2>
                <img src={plots.comparison} alt={t("results.comparisonAlt")} />
              </div>
              <div className="card">
                <h2>{t("results.decomposition")}</h2>
                <img
                  src={plots.decomposition}
                  alt={t("results.decompositionAlt")}
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
