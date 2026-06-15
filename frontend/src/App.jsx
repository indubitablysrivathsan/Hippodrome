import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import VenuesPage from "./pages/VenuesPage";
import MeetingsPage from "./pages/MeetingsPage";
import RacesPage from "./pages/RacesPage";
import RunnersPage from "./pages/RunnersPage";
import HorsesPage from "./pages/HorsesPage";
import JockeysPage from "./pages/JockeysPage";
import TrainersPage from "./pages/TrainersPage";
import SchemaPage from "./pages/SchemaPage";

export default function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/venues">Venues</Link>
        <Link to="/meetings">Meetings</Link>
        <Link to="/races">Races</Link>
        <Link to="/runners">Runners</Link>
        <Link to="/horses">Horses</Link>
        <Link to="/jockeys">Jockeys</Link>
        <Link to="/trainers">Trainers</Link>
        <Link to="/schema">Schema</Link>
      </nav>

      <Routes>
        <Route path="/" element={<VenuesPage />} />
        <Route path="/venues" element={<VenuesPage />} />
        <Route path="/meetings" element={<MeetingsPage />} />
        <Route path="/races" element={<RacesPage />} />
        <Route path="/runners" element={<RunnersPage />} />
        <Route path="/horses" element={<HorsesPage />} />
        <Route path="/jockeys" element={<JockeysPage />} />
        <Route path="/trainers" element={<TrainersPage />} />
        <Route path="/schema" element={<SchemaPage />} />
      </Routes>
    </BrowserRouter>
  );
}