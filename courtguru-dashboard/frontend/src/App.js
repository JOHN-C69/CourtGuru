import React, { useState, useEffect, useCallback } from "react";
import "./App.css";

function App() {
  const [bets, setBets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [sortField, setSortField] = useState("edge");
  const [sortDir, setSortDir] = useState("desc");
  const [filterTournament, setFilterTournament] = useState("All");
  const [filterBetType, setFilterBetType] = useState("All");
  const [minEdge, setMinEdge] = useState(0);

  // Fetch bets from backend
  const fetchBets = useCallback(() => {
    fetch("http://localhost:8000/bets")
      .then((res) => res.json())
      .then((data) => {
        setBets(data.bets);
        setLastUpdated(new Date(data.last_updated));
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error:", err);
        setLoading(false);
      });
  }, []);

  // Fetch on load + every 30 seconds
  useEffect(() => {
    fetchBets();
    const interval = setInterval(fetchBets, 30000);
    return () => clearInterval(interval);
  }, [fetchBets]);

  // Sorting
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const sortArrow = (field) => {
    if (sortField !== field) return "";
    return sortDir === "asc" ? " ▲" : " ▼";
  };

  // Get unique values for filter dropdowns
  const tournaments = ["All", ...new Set(bets.map((b) => b.tournament))];
  const betTypes = ["All", ...new Set(bets.map((b) => b.bet_type))];

  // Filter + sort
  const filteredBets = bets
    .filter((b) => filterTournament === "All" || b.tournament === filterTournament)
    .filter((b) => filterBetType === "All" || b.bet_type === filterBetType)
    .filter((b) => b.edge >= minEdge / 100)
    .sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];
      if (typeof valA === "string") {
        return sortDir === "asc"
          ? valA.localeCompare(valB)
          : valB.localeCompare(valA);
      }
      return sortDir === "asc" ? valA - valB : valB - valA;
    });

  const confidenceColor = (level) => {
    if (level === "high") return "#4ade80";
    if (level === "medium") return "#facc15";
    return "#f87171";
  };

  return (
    <div className="app">
      <header className="header">
        <div className="title-row">
          <h1>🎾 CourtGuru +EV Finder</h1>
          {lastUpdated && (
            <span className="updated">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>

        <div className="stats-row">
          <div className="stat-card">
            <span className="stat-value">{filteredBets.length}</span>
            <span className="stat-label">+EV Bets</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">
              {filteredBets.length > 0
                ? (
                    Math.max(...filteredBets.map((b) => b.edge)) * 100
                  ).toFixed(1) + "%"
                : "—"}
            </span>
            <span className="stat-label">Best Edge</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">
              {filteredBets.length > 0
                ? (
                    (filteredBets.reduce((s, b) => s + b.edge, 0) /
                      filteredBets.length) *
                    100
                  ).toFixed(1) + "%"
                : "—"}
            </span>
            <span className="stat-label">Avg Edge</span>
          </div>
        </div>

        <div className="filters">
          <select
            value={filterTournament}
            onChange={(e) => setFilterTournament(e.target.value)}
          >
            {tournaments.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>

          <select
            value={filterBetType}
            onChange={(e) => setFilterBetType(e.target.value)}
          >
            {betTypes.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>

          <div className="edge-filter">
            <label>Min Edge: {minEdge}%</label>
            <input
              type="range"
              min="0"
              max="20"
              value={minEdge}
              onChange={(e) => setMinEdge(Number(e.target.value))}
            />
          </div>
        </div>
      </header>

      {loading ? (
        <p className="status">Loading bets...</p>
      ) : filteredBets.length === 0 ? (
        <p className="status">No +EV bets match your filters.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th onClick={() => handleSort("match")}>
                Match{sortArrow("match")}
              </th>
              <th onClick={() => handleSort("tournament")}>
                Tournament{sortArrow("tournament")}
              </th>
              <th onClick={() => handleSort("pick")}>
                Pick{sortArrow("pick")}
              </th>
              <th onClick={() => handleSort("odds")}>
                Odds{sortArrow("odds")}
              </th>
              <th onClick={() => handleSort("model_prob")}>
                Model %{sortArrow("model_prob")}
              </th>
              <th onClick={() => handleSort("implied_prob")}>
                Implied %{sortArrow("implied_prob")}
              </th>
              <th onClick={() => handleSort("edge")}>
                Edge{sortArrow("edge")}
              </th>
              <th onClick={() => handleSort("kelly_fraction")}>
                Kelly{sortArrow("kelly_fraction")}
              </th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {filteredBets.map((bet) => (
              <tr key={bet.id}>
                <td className="match-cell">
                  {bet.match}
                  <span className="surface">{bet.surface}</span>
                </td>
                <td>{bet.tournament}</td>
                <td className="pick-cell">{bet.pick}</td>
                <td>{bet.odds}</td>
                <td>{(bet.model_prob * 100).toFixed(1)}%</td>
                <td>{(bet.implied_prob * 100).toFixed(1)}%</td>
                <td className="edge">{(bet.edge * 100).toFixed(1)}%</td>
                <td>{(bet.kelly_fraction * 100).toFixed(1)}%</td>
                <td>
                  <span
                    className="confidence-dot"
                    style={{ background: confidenceColor(bet.confidence) }}
                  ></span>
                  {bet.confidence}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;