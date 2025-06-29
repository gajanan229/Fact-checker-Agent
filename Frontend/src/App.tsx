import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import WorkbenchPage from './pages/WorkbenchPage';
import './index.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen verifyp-gradient">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/workbench" element={<WorkbenchPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;