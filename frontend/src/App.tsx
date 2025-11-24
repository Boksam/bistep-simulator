import { useState, useEffect } from "react";
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
import InfoTooltip from "./components/InfoTooltip";

// --- TypeScript Interfaces ---
interface SimulationDataPoint {
  timestamp: string;
  value: number;
}

interface Plots {
  decomposition: string;
  comparison: string;
}

interface ModelParameters {
  trend_coefficients?: number[];
  noise_std?: number;
  noise_type?: string;
  seasonality_strength?: number;
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
  const [modelParameters, setModelParameters] =
    useState<ModelParameters | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formula, setFormula] = useState<string | null>(null);

  // Synthetic Generator State
  const [mode, setMode] = useState<"historical" | "synthetic">("historical");
  const [patternType, setPatternType] = useState("polynomial");
  const [polyDegree, setPolyDegree] = useState(1);
  const [polyCoeffs, setPolyCoeffs] = useState<number[]>([0, 1]);
  const [expParams, setExpParams] = useState({
    scale: 1.0,
    rate: 0.5,
    offset: 0.0,
  });
  const [stepParams, setStepParams] = useState({
    min_val: 0,
    max_val: 10,
    step_position: 5.0,
    transition_speed: 2.0,
  });
  const [noiseStd, setNoiseStd] = useState(0.0);

  // Update poly coeffs when degree changes
  useEffect(() => {
    if (polyCoeffs.length !== polyDegree + 1) {
      const newCoeffs = new Array(polyDegree + 1).fill(0);
      for (let i = 0; i < Math.min(newCoeffs.length, polyCoeffs.length); i++) {
        newCoeffs[newCoeffs.length - 1 - i] =
          polyCoeffs[polyCoeffs.length - 1 - i];
      }
      setPolyCoeffs(newCoeffs);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polyDegree]);

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
    setModelParameters(null);
    setFormula(null);

    try {
      let response;
      if (mode === "synthetic") {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let params: any = { noise_std: noiseStd };
        if (patternType === "polynomial") {
          params.coefficients = polyCoeffs;
        } else if (patternType === "exponential") {
          params = { ...params, ...expParams };
        } else if (patternType === "step") {
          params = { ...params, ...stepParams };
        }

        response = await axios.post(
          `${import.meta.env.VITE_API_BASE_URL}/generate-synthetic`,
          {
            pattern_type: patternType,
            parameters: params,
            start_date: startDate,
            end_date: endDate,
            interval: interval,
          }
        );
        setSimulationData(response.data);
      } else {
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
          setModelParameters(response.data.model_parameters);
        }
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

  // Auto-generate for synthetic mode
  useEffect(() => {
    if (mode === "synthetic") {
      const timer = setTimeout(() => {
        handleRunSimulation();
      }, 500);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    mode,
    patternType,
    polyDegree,
    polyCoeffs,
    expParams,
    stepParams,
    noiseStd,
    startDate,
    endDate,
    interval,
  ]);

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
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "1rem",
          }}
        >
          <h2 style={{ margin: 0 }}>{t("configuration.title")}</h2>
          <div
            className="mode-toggle"
            style={{ display: "flex", alignItems: "center", gap: "1rem" }}
          >
            <span
              style={{
                fontWeight: mode === "historical" ? "bold" : "normal",
                color: mode === "historical" ? "#007bff" : "#666",
                cursor: "pointer",
              }}
              onClick={() => setMode("historical")}
            >
              {t("mode.historical")}
            </span>
            <div
              onClick={() =>
                setMode(mode === "historical" ? "synthetic" : "historical")
              }
              style={{
                width: "48px",
                height: "24px",
                backgroundColor: mode === "synthetic" ? "#007bff" : "#ccc",
                borderRadius: "12px",
                position: "relative",
                cursor: "pointer",
                transition: "background-color 0.3s",
              }}
            >
              <div
                style={{
                  width: "20px",
                  height: "20px",
                  backgroundColor: "white",
                  borderRadius: "50%",
                  position: "absolute",
                  top: "2px",
                  left: mode === "synthetic" ? "26px" : "2px",
                  transition: "left 0.3s",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
                }}
              />
            </div>
            <span
              style={{
                fontWeight: mode === "synthetic" ? "bold" : "normal",
                color: mode === "synthetic" ? "#007bff" : "#666",
                cursor: "pointer",
              }}
              onClick={() => setMode("synthetic")}
            >
              {t("mode.synthetic")}
            </span>
          </div>
        </div>

        <div className="control-grid">
          {/* Slot 1: Sensor Type OR Pattern Type */}
          <div>
            <label htmlFor="type-select">
              {mode === "historical"
                ? t("configuration.sensorType")
                : t("synthetic.patternType")}
            </label>
            {mode === "historical" ? (
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
            ) : (
              <select
                value={patternType}
                onChange={(e) => setPatternType(e.target.value)}
              >
                <option value="polynomial">
                  {t("synthetic.patterns.polynomial")}
                </option>
                <option value="exponential">
                  {t("synthetic.patterns.exponential")}
                </option>
                <option value="step">{t("synthetic.patterns.step")}</option>
              </select>
            )}
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

        {/* Synthetic Mode Specific Controls */}
        {mode === "synthetic" && (
          <div
            className="synthetic-controls"
            style={{
              marginTop: "1.5rem",
              borderTop: "1px solid #eee",
              paddingTop: "1.5rem",
            }}
          >
            {patternType === "polynomial" && (
              <div className="control-group">
                <label>{t("synthetic.degree")}</label>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "row",
                    gap: "0.5rem",
                    marginBottom: "1rem",
                  }}
                >
                  {[0, 1, 2].map((d) => (
                    <button
                      key={d}
                      onClick={() => setPolyDegree(d)}
                      style={{
                        padding: "0.25rem 0.75rem",
                        borderRadius: "4px",
                        border: "1px solid #ccc",
                        background: polyDegree === d ? "#007bff" : "#fff",
                        color: polyDegree === d ? "white" : "#333",
                        cursor: "pointer",
                      }}
                    >
                      {d}
                    </button>
                  ))}
                </div>
                <div className="coefficients">
                  <label>{t("synthetic.coefficients")}</label>
                  {polyCoeffs.map((c, i) => (
                    <div
                      key={i}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                        marginBottom: "0.5rem",
                      }}
                    >
                      <span style={{ width: "3rem" }}>x^{polyDegree - i}</span>
                      <input
                        type="range"
                        min="-10"
                        max="10"
                        step="0.1"
                        value={c}
                        onChange={(e) => {
                          const newCoeffs = [...polyCoeffs];
                          newCoeffs[i] = parseFloat(e.target.value);
                          setPolyCoeffs(newCoeffs);
                        }}
                        style={{ flex: 1 }}
                      />
                      <input
                        type="number"
                        value={c}
                        onChange={(e) => {
                          const newCoeffs = [...polyCoeffs];
                          newCoeffs[i] = parseFloat(e.target.value);
                          setPolyCoeffs(newCoeffs);
                        }}
                        style={{ width: "4rem" }}
                      />
                    </div>
                  ))}
                </div>
                <div
                  style={{
                    marginTop: "1rem",
                    padding: "0.5rem",
                    background: "#f8f9fa",
                    borderRadius: "4px",
                  }}
                >
                  <BlockMath
                    math={`y = ${polyCoeffs
                      .map((c, i) => `${c}x^{${polyDegree - i}}`)
                      .join(" + ")}`}
                  />
                </div>
              </div>
            )}

            {patternType === "exponential" && (
              <div className="control-group">
                {Object.entries(expParams).map(([key, val]) => (
                  <div
                    key={key}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      marginBottom: "0.5rem",
                    }}
                  >
                    <span
                      style={{ width: "4rem", textTransform: "capitalize" }}
                    >
                      {t(`synthetic.params.${key}`)}
                    </span>
                    <input
                      type="range"
                      min={key === "rate" ? "-2" : "-10"}
                      max={key === "rate" ? "2" : "20"}
                      step="0.1"
                      value={val}
                      onChange={(e) =>
                        setExpParams({
                          ...expParams,
                          [key]: parseFloat(e.target.value),
                        })
                      }
                      style={{ flex: 1 }}
                    />
                    <input
                      type="number"
                      value={val}
                      onChange={(e) =>
                        setExpParams({
                          ...expParams,
                          [key]: parseFloat(e.target.value),
                        })
                      }
                      style={{ width: "4rem" }}
                    />
                  </div>
                ))}
                <div
                  style={{
                    marginTop: "1rem",
                    padding: "0.5rem",
                    background: "#f8f9fa",
                    borderRadius: "4px",
                  }}
                >
                  <BlockMath
                    math={`y = ${expParams.scale} \\cdot e^{${expParams.rate}x} + ${expParams.offset}`}
                  />
                </div>
              </div>
            )}

            {patternType === "step" && (
              <div className="control-group">
                {Object.entries(stepParams).map(([key, val]) => (
                  <div
                    key={key}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      marginBottom: "0.5rem",
                    }}
                  >
                    <span
                      style={{ width: "8rem", textTransform: "capitalize" }}
                    >
                      {t(`synthetic.params.${key}`)}
                    </span>
                    <input
                      type="range"
                      min={key.includes("val") ? "-10" : "0"}
                      max={key.includes("val") ? "20" : "10"}
                      step="0.1"
                      value={val}
                      onChange={(e) =>
                        setStepParams({
                          ...stepParams,
                          [key]: parseFloat(e.target.value),
                        })
                      }
                      style={{ flex: 1 }}
                    />
                    <input
                      type="number"
                      value={val}
                      onChange={(e) =>
                        setStepParams({
                          ...stepParams,
                          [key]: parseFloat(e.target.value),
                        })
                      }
                      style={{ width: "4rem" }}
                    />
                  </div>
                ))}
                <div
                  style={{
                    marginTop: "1rem",
                    padding: "0.5rem",
                    background: "#f8f9fa",
                    borderRadius: "4px",
                  }}
                >
                  <BlockMath
                    math={`y = ${stepParams.min_val} + \\frac{${stepParams.max_val} - ${stepParams.min_val}}{1 + e^{-${stepParams.transition_speed}(x - ${stepParams.step_position})}}`}
                  />
                </div>
              </div>
            )}

            <div
              style={{
                marginTop: "1rem",
                borderTop: "1px solid #eee",
                paddingTop: "1rem",
              }}
            >
              <label>{t("synthetic.noiseLevel")}</label>
              <div
                style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
              >
                <input
                  type="range"
                  min="0"
                  max="5"
                  step="0.1"
                  value={noiseStd}
                  onChange={(e) => setNoiseStd(parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span>{noiseStd}</span>
              </div>
            </div>
          </div>
        )}

        {mode === "historical" && sensorType === "anode_lifetime" && (
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

        {mode === "historical" && (
          <button onClick={handleRunSimulation} disabled={loading}>
            {loading ? t("buttons.generating") : t("buttons.runSimulation")}
          </button>
        )}
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
                {mode === "synthetic"
                  ? `${t("mode.synthetic")} ${t("results.simulationResults")}`
                  : `${getSensorDisplayName(sensorType)} ${t(
                      "results.simulationResults"
                    )}`}
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
                    name={
                      mode === "synthetic"
                        ? "Generated Data"
                        : `${t("results.simulated")} ${getSensorDisplayName(
                            sensorType
                          )}`
                    }
                    stroke="#8884d8"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {modelParameters && (
            <div className="card">
              <h2>{t("results.modelParameters")}</h2>
              <ul>
                {modelParameters.trend_coefficients && (
                  <li>
                    <strong>{t("results.trendCoefficients")}:</strong>
                    <InfoTooltip
                      text={t("results.trendCoefficientsTooltip")}
                    />{" "}
                    {modelParameters.trend_coefficients
                      .map((c) => c.toFixed(4))
                      .join(", ")}
                  </li>
                )}
                {modelParameters.noise_std !== undefined && (
                  <li>
                    <strong>{t("results.noiseStd")}:</strong>
                    <InfoTooltip text={t("results.noiseStdTooltip")} />{" "}
                    {modelParameters.noise_std.toFixed(4)}
                  </li>
                )}
                {modelParameters.seasonality_strength !== undefined && (
                  <li>
                    <strong>{t("results.seasonalityStrength")}:</strong>
                    <InfoTooltip
                      text={t("results.seasonalityStrengthTooltip")}
                    />{" "}
                    {modelParameters.seasonality_strength.toFixed(4)}
                  </li>
                )}
              </ul>
            </div>
          )}

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
