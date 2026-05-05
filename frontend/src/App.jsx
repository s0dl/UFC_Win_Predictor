import React from "react";
import { useState } from "react";
import PredictView from "./components/PredictView.jsx";

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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

      <PredictView apiUrl={apiUrl} />
    </main>
  );
}

export default App;
