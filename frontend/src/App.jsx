import React from "react";
import { useState } from "react";
import NextEventEdges from "./components/NextEventEdges.jsx";
import PredictView from "./components/PredictView.jsx";

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const METHODOLOGY_URL = import.meta.env.VITE_METHODOLOGY_URL || "https://github.com/s0dl/UFC_Win_Predictor";

function App() {
  const [apiUrl] = useState(DEFAULT_API_URL);

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Fight night model</p>
          <h1>UFC Fight Predictor</h1>
        </div>
      </header>

      <NextEventEdges apiUrl={apiUrl} />
      <PredictView apiUrl={apiUrl} />

      <footer className="page-footer">
        <a href={METHODOLOGY_URL}>Methodology</a>
      </footer>
    </main>
  );
}

export default App;
