import { useState } from "react";
import { runners } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

const SUBS = [
  { key: "detail", label: "Detail" },
  { key: "acceptance", label: "Acceptance" },
  { key: "declaration", label: "Declaration" },
  { key: "equipment", label: "Equipment" },
  { key: "equipmentChanges", label: "Equipment Changes" },
  { key: "jockeyChange", label: "Jockey Change" },
];

export default function RunnersPage() {
  const apis = {
    detail: useApi(runners.get),
    acceptance: useApi(runners.acceptance),
    declaration: useApi(runners.declaration),
    equipment: useApi(runners.equipment),
    equipmentChanges: useApi(runners.equipmentChanges),
    jockeyChange: useApi(runners.jockeyChange),
  };

  const [raceId, setRaceId] = useState("");
  const [horseId, setHorseId] = useState("");

  return (
    <div>
      <h2>Runners</h2>
      <p style={{ color: "#555", fontSize: 13 }}>All endpoints require Race ID + Horse ID.</p>

      <div style={{ marginBottom: 12 }}>
        <Field label="Race ID" value={raceId} onChange={setRaceId} placeholder="e.g. 1" />
        <Field label="Horse ID" value={horseId} onChange={setHorseId} placeholder="e.g. 1" />
      </div>

      {SUBS.map(({ key, label }) => (
        <Section key={key} title={label}>
          <Btn onClick={() => apis[key].run(raceId, horseId)} disabled={!raceId || !horseId}>
            Fetch {label}
          </Btn>
          <Status {...apis[key]} />
          {apis[key].data && <Table rows={toRows(apis[key].data)} />}
        </Section>
      ))}
    </div>
  );
}
