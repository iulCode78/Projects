// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SpotifyForm from './spotifyform';
import RecommendedSongs from './RecommendedSongs';
import Header from './header';
import Footer from './footer';

function App() {
  return (
    <Router>
      <Header /> {/* Header is placed here to be shown on all pages */}
      <Routes>
        <Route path="/" element={<SpotifyForm />} />
        <Route path="/recommendations" element={<RecommendedSongs />} />
      </Routes>
     
      <Footer/>
    </Router>
  );
}

export default App;


